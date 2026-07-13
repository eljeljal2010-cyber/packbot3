import os
import json
import re
import statistics
import asyncio
import itertools
from datetime import timedelta, timezone, datetime
from typing import Optional
from pathlib import Path
from urllib.parse import quote
import discord
from discord import app_commands
from discord.ext import commands, tasks
from vinted import (
    VintedClient,
    VintedRateLimitError,
    VintedNetworkError,
    VintedAPIError,
    VintedAuthError,
    VintedError,
)
from groq import Groq

# --- Configuration de base ---
intents = discord.Intents.default()
intents.message_content = True  # nécessaire pour lire le texte des messages (chat IA)
bot = commands.Bot(command_prefix="!", intents=intents)

GUILD_ID = os.environ.get("GUILD_ID")
CHAT_CHANNEL_ID = os.environ.get("CHAT_CHANNEL_ID")  # si défini, le chat IA ne répond que dans ce salon

# --- Configuration du chat IA (Groq, gratuit) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"⚠️ Impossible d'initialiser le client Groq : {e}")
else:
    print(
        "⚠️ ATTENTION : GROQ_API_KEY n'est pas défini. Le bot démarre quand même (/poster, /estimer, "
        "/recherche, les alertes restent fonctionnels), mais le chat IA et /description répondront "
        "avec un message d'erreur clair tant que la variable n'est pas ajoutée sur Railway."
    )
# ⚠️ llama-3.3-70b-versatile est déprécié par Groq (arrêt prévu le 16/08/2026) et
# llama-4-scout-17b-16e-instruct l'est aussi (arrêt prévu le 17/07/2026) : on utilise
# directement les remplacements officiels recommandés par Groq, redéfinissables via
# variable d'environnement au cas où Groq change encore d'avis d'ici là.
MODELE_REDACTION = os.environ.get("GROQ_REDACTION_MODEL", "openai/gpt-oss-120b")
MODELE_VISION = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")  # gpt-oss-120b ne gère pas les images
# groq/compound = système agentique qui ajoute recherche web temps réel + exécution de code
# par-dessus des modèles puissants (GPT-OSS 120B, Llama 4/3.3) : plus capable ET informé en direct.
MODELE_CONVERSATION = os.environ.get("GROQ_CONVERSATION_MODEL", "groq/compound")
# Modèle rapide/économique dédié aux résumés de mémoire (pas besoin de puissance ici).
MODELE_RESUME = os.environ.get("GROQ_RESUME_MODEL", "llama-3.1-8b-instant")


def _options_raisonnement(modele: str) -> dict:
    """openai/gpt-oss-* et qwen3.x sont des modèles 'raisonneurs' : ils réfléchissent dans un champ
    caché avant de répondre. On vise un effort 'medium'/'low' (plutôt que le minimum) pour de
    meilleures réponses, tout en gardant les filets de sécurité existants (retry en texte seul,
    retry si réponse vide) qui absorbent le risque que le raisonnement consomme trop de budget."""
    modele_lower = modele.lower()
    if "gpt-oss" in modele_lower:
        return {"reasoning_effort": "medium"}
    if "qwen3" in modele_lower:
        return {"reasoning_effort": "low"}
    return {}


class GroqNonConfigure(Exception):
    """Levée quand GROQ_API_KEY n'est pas configurée — message clair plutôt qu'un AttributeError opaque."""
    pass


def _verifier_groq_disponible():
    if groq_client is None:
        raise GroqNonConfigure(
            "La clé GROQ_API_KEY n'est pas configurée sur le serveur (variable d'environnement manquante)."
        )


def _erreur_ia_lisible(e: Exception) -> str:
    """Traduit les erreurs Groq les plus courantes en message compréhensible, plutôt que de balancer
    le JSON brut de l'API à l'utilisateur. Le fameux '413 Request Entity Too Large' de Groq n'est en
    réalité PAS une histoire de message trop long : c'est une limite de débit (tokens/minute) du
    compte — donc pas la peine de le présenter comme une erreur du message envoyé."""
    if isinstance(e, GroqNonConfigure):
        return "🔑 Le chat IA n'est pas configuré (clé GROQ_API_KEY manquante) — préviens l'admin du serveur."
    texte = str(e)
    texte_lower = texte.lower()
    if "413" in texte or "request_too_large" in texte_lower:
        return (
            "🚦 Limite de débit IA atteinte sur le compte Groq (trop de tokens demandés en peu de temps, "
            "pas un souci avec ton message). Réessaie dans une minute, ça repasse tout seul."
        )
    if "429" in texte or "rate_limit" in texte_lower:
        return "🚦 Trop de demandes IA en même temps, réessaie dans quelques secondes."
    if "401" in texte or "invalid_api_key" in texte_lower:
        return "🔑 Problème de clé API Groq — préviens l'admin du serveur."
    return f"Erreur inattendue : `{texte[:300]}`"


async def _appeler_groq(**kwargs):
    """Appelle Groq en tentant d'abord avec les options de raisonnement réduites, puis se rabat sur
    un appel simple si le modèle/l'API ne supporte pas ces paramètres."""
    _verifier_groq_disponible()
    modele = kwargs.get("model", "")
    options = _options_raisonnement(modele)
    try:
        return await asyncio.to_thread(groq_client.chat.completions.create, **{**kwargs, **options})
    except Exception:
        if options:
            return await asyncio.to_thread(groq_client.chat.completions.create, **kwargs)
        raise
HISTORIQUE_MAX = 16  # nombre de messages bruts gardés en mémoire par salon
HISTORIQUE_SEUIL_RESUME = 20  # au-delà, on condense les plus anciens en résumé
HISTORIQUE_GARDE_APRES_RESUME = 10  # nombre de messages bruts gardés après condensation
RESUME_MAX_CARACTERES = 2000  # taille max du résumé cumulé, pour ne pas gonfler indéfiniment
historique_conversations = {}  # {channel_id: [ {"role": ..., "content": ...}, ... ]}
resume_conversations = {}  # {channel_id: "résumé condensé des échanges plus anciens"}


async def _maj_resume_si_necessaire(channel_id):
    """Si l'historique brut devient trop long, condense les messages les plus anciens en un résumé
    (via un modèle rapide/économique) pour que le bot 'se souvienne' de loin sans faire exploser le
    nombre de tokens envoyés à chaque appel."""
    historique = historique_conversations.get(channel_id, [])
    if len(historique) <= HISTORIQUE_SEUIL_RESUME:
        return

    a_condenser = historique[:-HISTORIQUE_GARDE_APRES_RESUME]
    texte_a_condenser = "\n".join(f"{m['role']} : {m['content']}" for m in a_condenser)

    try:
        reponse = await _appeler_groq(
            model=MODELE_RESUME,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Condense cette portion de conversation Discord en un résumé dense de quelques "
                        "phrases, en gardant uniquement les informations utiles pour la suite (préférences "
                        "de la personne, articles évoqués, prix discutés, décisions prises). Pas de "
                        "formule d'introduction, juste les faits utiles."
                    ),
                },
                {"role": "user", "content": texte_a_condenser},
            ],
        )
        nouveau_bout = (reponse.choices[0].message.content or "").strip()
        if nouveau_bout:
            ancien_resume = resume_conversations.get(channel_id, "")
            fusion = f"{ancien_resume}\n{nouveau_bout}".strip()
            resume_conversations[channel_id] = fusion[-RESUME_MAX_CARACTERES:]
    except Exception as e:
        print(f"[chat] échec du résumé de mémoire : {e}")

    historique_conversations[channel_id] = historique[-HISTORIQUE_GARDE_APRES_RESUME:]

STYLES = {
    "marseillais": {
        "nom": "🌞 Marseillais",
        "prompt": (
            "Tu es un assistant IA sur un serveur Discord francophone dédié à l'achat/revente d'articles "
            "d'occasion. Tu as un accent et un parler marseillais bien marqué : utilise naturellement des "
            "expressions typiques comme 'vé', 'peuchère', 'fada', 'dégun', 'cong', 'oh putaing', "
            "'je te le dis', 'quiche' etc., et termine parfois tes phrases par 'quoi' ou 'tu vois'. "
            "Reste toujours sympathique, clair et utile malgré l'accent — on doit comprendre facilement "
            "tes réponses. Sois concis (quelques phrases, sauf si on te demande plus de détails)."
        ),
        "intro": "Vé, c'est moi qui commande ici maintenant, tu vois ! Dis-moi tout, fada 😄",
    },
    "parigot": {
        "nom": "🥖 Parigot",
        "prompt": (
            "Tu es un assistant IA sur un serveur Discord francophone dédié à l'achat/revente d'articles "
            "d'occasion. Tu parles comme un pur Parigot : ton direct, un poil cash, débit rapide, avec des "
            "expressions typiques ('wesh', 'grave', 'sérieux', 'ouais enfin', 'tqt', 'flemme', 'ça le fait') "
            "et une pointe d'humour parisien. Reste toujours utile et clair malgré le ton familier. "
            "Sois concis (quelques phrases, sauf si on te demande plus de détails)."
        ),
        "intro": "Wesh, on est repartis, grave ! Vas-y balance ta question 🥖",
    },
    "serieux": {
        "nom": "💼 Sérieux (SAV)",
        "prompt": (
            "Tu es l'assistant du service client (SAV) sur un serveur Discord dédié à l'achat/revente "
            "d'articles d'occasion. Adopte un ton professionnel, courtois et rassurant, sans familiarité "
            "ni accent particulier. Sois précis, structuré, et va droit au but tout en restant chaleureux. "
            "Si la question concerne un litige, un remboursement ou un problème de commande, pose des "
            "questions de clarification si nécessaire avant de proposer une solution."
        ),
        "intro": "Bonjour, je passe en mode support client. Je reste à votre disposition pour toute question. 💼",
    },
    "hype": {
        "nom": "🔥 Vendeur hype",
        "prompt": (
            "Tu es un assistant IA sur un serveur Discord dédié à l'achat/revente d'articles d'occasion. "
            "Tu as l'énergie d'un vendeur ultra motivé et enthousiaste : exclamations, punchlines courtes, "
            "tu mets en avant les bonnes affaires et donnes envie d'agir vite ('à ce prix-là ça va pas "
            "durer !'). Reste honnête, jamais mensonger sur les prix ou l'état des articles. Sois concis."
        ),
        "intro": "🔥 C'est parti, mode hype activé ! On va faire des affaires en OR aujourd'hui 💰",
    },
    "cash": {
        "nom": "🧊 Négociateur cash",
        "prompt": (
            "Tu es un assistant IA sur un serveur Discord dédié à l'achat/revente d'articles d'occasion. "
            "Ton ton est froid, direct et sans détour, façon négociateur qui ne perd pas de temps en "
            "formules de politesse superflues. Tu vas à l'essentiel, tu donnes des chiffres et des faits, "
            "sans être désagréable pour autant. Sois très concis."
        ),
        "intro": "Ok. Mode direct activé. Pose ta question, j'irai droit au but.",
    },
}

STYLE_PAR_DEFAUT = "marseillais"
style_actuel = {}  # {channel_id: style_key} — style courant du chat IA par salon


def _style_du_salon(channel_id) -> str:
    return style_actuel.get(channel_id, STYLE_PAR_DEFAUT)


# --- Tons disponibles pour le générateur de description d'annonce (/annonce) ---
TONS_ANNONCE = {
    "accrocheur": "dynamique et percutant, orienté vente, avec quelques emojis pertinents sans excès",
    "sobre": "neutre et factuel, précis, avec très peu voire pas d'emojis, focus sur les caractéristiques concrètes",
    "fun": "familier, complice avec l'acheteur, avec de l'humour léger et des emojis",
}

TONS_ANNONCE_LABELS = {
    "accrocheur": "✨ Accrocheur",
    "sobre": "📝 Sobre et factuel",
    "fun": "😄 Fun / familier",
    "personnalite": "🎭 Ma personnalité actuelle",
}


# ============================================================
#  Fonctions utilitaires
# ============================================================

def _champ(obj, cle, defaut=None):
    """Récupère un champ que 'obj' soit un dict (JSON brut) ou un objet avec attributs."""
    if isinstance(obj, dict):
        return obj.get(cle, defaut)
    return getattr(obj, cle, defaut)


def _prix_de(obj):
    """Extrait un prix flottant, que le champ 'price' soit un nombre, une chaîne, ou un dict {'amount': ...}."""
    p = _champ(obj, "price")
    if isinstance(p, dict):
        p = p.get("amount")
    try:
        return float(p)
    except (TypeError, ValueError):
        return None


def _photo_de(obj):
    """Récupère l'URL de la photo principale d'une annonce, quel que soit le format renvoyé."""
    photo = _champ(obj, "photo")
    if isinstance(photo, dict):
        return photo.get("url") or photo.get("full_size_url")
    if isinstance(photo, list) and photo:
        premiere = photo[0]
        if isinstance(premiere, dict):
            return premiere.get("url") or premiere.get("full_size_url")
        if isinstance(premiere, str):
            return premiere
    if isinstance(photo, str):
        return photo
    return None


def _favoris_de(obj):
    return _champ(obj, "favourite_count", 0) or 0


def _marge_cible_par_defaut(prix_achat: float) -> float:
    """
    Marge visée (en %) adaptée au prix d'achat, dans une logique de flip :
    plus l'article est bon marché, plus on vise un multiplicateur élevé.
    """
    if prix_achat <= 15:
        return 200.0  # viser le triple
    elif prix_achat <= 40:
        return 100.0  # viser le double
    else:
        return 50.0  # viser +50%


def _filtrer_valeurs_extremes(prix_valides):
    """
    Retire les prix aberrants (méthode IQR) qui fausseraient la moyenne,
    par exemple une annonce à 3€ perdue au milieu d'annonces à 40€.
    Retourne (liste_filtrée, nombre_exclu).
    """
    if len(prix_valides) < 4:
        return prix_valides, 0

    trie = sorted(prix_valides)
    q1, q3 = statistics.quantiles(trie, n=4)[0], statistics.quantiles(trie, n=4)[2]
    iqr = q3 - q1
    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr

    filtres = [p for p in prix_valides if borne_basse <= p <= borne_haute]
    if not filtres:  # sécurité, ne devrait pas arriver
        return prix_valides, 0
    return filtres, len(prix_valides) - len(filtres)


# ============================================================
#  Vue : bouton "Accès au lien" (masqué jusqu'au clic)
# ============================================================

class LinkButtonView(discord.ui.View):
    """Bouton-lien natif Discord (style=link) : contrairement à un bouton avec callback, celui-ci est
    entièrement géré côté Discord et ne passe jamais par le bot pour fonctionner. Il continue donc de
    marcher indéfiniment, même après un redémarrage ou un redéploiement — plus jamais d'« Échec de
    l'interaction » sur un vieux message. Le lien reste discret : personne ne voit l'URL en clair dans
    le message, seul le clic y donne accès."""

    def __init__(self, lien: str):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="Accès au lien", style=discord.ButtonStyle.link, emoji="🔗", url=lien))


# ============================================================
#  Démarrage du bot
# ============================================================

@bot.event
async def on_ready():
    try:
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
            # Supprime les anciennes commandes globales en double.
            bot.tree.clear_commands(guild=None)
            await bot.tree.sync()
        else:
            synced = await bot.tree.sync()
        print(f"{len(synced)} commande(s) synchronisée(s) sur le serveur.")
    except Exception as e:
        print(f"Erreur de synchronisation : {e}")

    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️ Rappel : GROQ_API_KEY toujours absent, le chat IA et /description restent désactivés.")

    # Vues persistantes : ré-enregistrées à chaque démarrage pour que les boutons "Lancer une
    # estimation" / "Générer la description" publiés avant un redémarrage du bot restent cliquables
    # (sans ça, Discord affiche une erreur sur les anciens boutons après chaque redéploiement Railway).
    bot.add_view(EstimerIntroView())
    bot.add_view(AnnonceIntroView())
    bot.add_view(RechercheIntroView())
    bot.add_view(SuiviView())

    if not verifier_alertes.is_running():
        verifier_alertes.start()

    print(f"Connecté en tant que {bot.user} — bot prêt.")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Filet de sécurité global : évite le message opaque 'L'application n'a pas répondu' de Discord
    si une commande plante de façon inattendue, et logge l'erreur côté serveur pour diagnostic."""
    print(f"[erreur commande] /{interaction.command.name if interaction.command else '?'} : {error!r}")
    message = "❌ Une erreur inattendue est survenue. Réessaie, et si ça persiste, préviens l'admin du serveur."
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except discord.HTTPException:
        pass


# ============================================================
#  Commande /poster
# ============================================================

@bot.tree.command(name="poster", description="Crée une annonce avec titre, texte encadré et bouton lien caché")
@app_commands.describe(
    titre="Le titre affiché en haut de l'annonce",
    texte="Le texte principal (affiché dans le cadre)",
    lien="Le lien qui sera révélé uniquement au clic (visible seulement par la personne qui clique)"
)
async def poster(interaction: discord.Interaction, titre: str, texte: str, lien: str):
    if not lien.startswith(("http://", "https://")):
        await interaction.response.send_message(
            "❌ Le lien doit commencer par `http://` ou `https://`.", ephemeral=True
        )
        return
    embed = discord.Embed(title=titre[:256], description=texte[:4096], color=discord.Color.blurple())
    embed.timestamp = discord.utils.utcnow()
    view = LinkButtonView(lien)
    await interaction.response.send_message(embed=embed, view=view)


# ============================================================
#  Chat IA (répond automatiquement dans le salon dédié)
# ============================================================

IMAGES_MAX_PAR_MESSAGE = 2  # limite par message pour contrôler le coût/la latence


def _images_jointes(message: discord.Message):
    extensions_image = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    images = [
        a for a in message.attachments
        if (a.content_type or "").startswith("image/") or a.filename.lower().endswith(extensions_image)
    ]
    return images[:IMAGES_MAX_PAR_MESSAGE]


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    dans_salon_dedie = CHAT_CHANNEL_ID and str(message.channel.id) == str(CHAT_CHANNEL_ID)
    mentionne = bot.user in message.mentions
    est_une_reponse_au_bot = (
        message.reference is not None
        and getattr(message.reference.resolved, "author", None) == bot.user
    )

    # Si un salon dédié est configuré, le bot ne répond QUE là (mention ou non).
    # Sinon (pas de salon configuré), il répond partout mais seulement si mentionné/répondu.
    if CHAT_CHANNEL_ID:
        if not dans_salon_dedie:
            return
    else:
        if not (mentionne or est_une_reponse_au_bot):
            return

    contenu = message.content
    for m in message.mentions:
        contenu = contenu.replace(f"<@{m.id}>", "").replace(f"<@!{m.id}>", "")
    contenu = contenu.strip()

    images = _images_jointes(message)
    if not contenu and not images:
        contenu = "Bonjour !"
    elif not contenu and images:
        contenu = "Regarde cette photo et donne-moi ton avis."

    async with message.channel.typing():
        historique = historique_conversations.setdefault(message.channel.id, [])
        # En mémoire, on garde une trace textuelle simple (pas l'image elle-même : les liens
        # d'attachement Discord peuvent expirer, autant garder l'historique léger et fiable).
        contenu_historique = contenu if not images else f"{contenu} [a envoyé {len(images)} photo(s)]"
        historique.append({"role": "user", "content": contenu_historique})
        del historique[:-HISTORIQUE_MAX]

        # Le tour courant peut inclure les images (uniquement pour cet appel, pas conservé en mémoire).
        if images:
            contenu_tour_courant = [{"type": "text", "text": contenu}]
            for img in images:
                contenu_tour_courant.append({"type": "image_url", "image_url": {"url": img.url}})
            modele_a_utiliser = MODELE_VISION
        else:
            contenu_tour_courant = contenu
            modele_a_utiliser = MODELE_CONVERSATION

        try:
            style_key = _style_du_salon(message.channel.id)
            messages_api = [{"role": "system", "content": STYLES[style_key]["prompt"]}]
            resume = resume_conversations.get(message.channel.id)
            if resume:
                messages_api.append({
                    "role": "system",
                    "content": f"Résumé des échanges précédents dans ce salon (pour mémoire) : {resume}",
                })
            messages_api += historique[:-1]  # les tours précédents, en texte simple
            messages_api.append({"role": "user", "content": contenu_tour_courant})  # le tour courant, avec photo si besoin

            taille_approx = sum(len(m["content"]) for m in messages_api if isinstance(m["content"], str))
            try:
                reponse = await _appeler_groq(
                    model=modele_a_utiliser,
                    max_tokens=1200,
                    messages=messages_api,
                )
            except Exception as e:
                # Le fameux "413" de Groq est en réalité une limite de débit (tokens/minute), pas un
                # souci de taille de message : on laisse un instant au quota de se libérer, on retente
                # avec un contexte minimal, et si possible sur un modèle moins gourmand que compound
                # (qui enchaîne plusieurs appels internes et consomme donc plus de tokens par requête).
                if "413" in str(e) or "request_too_large" in str(e).lower():
                    print(f"[chat] limite de débit Groq (~{taille_approx} caractères envoyés), nouvel essai : {e}")
                    resume_conversations.pop(message.channel.id, None)
                    historique_conversations[message.channel.id] = [{"role": "user", "content": contenu_historique}]
                    await asyncio.sleep(3)
                    modele_repli = MODELE_REDACTION if modele_a_utiliser == MODELE_CONVERSATION else modele_a_utiliser
                    try:
                        reponse = await _appeler_groq(
                            model=modele_repli,
                            max_tokens=1200,
                            messages=[
                                {"role": "system", "content": STYLES[style_key]["prompt"]},
                                {"role": "user", "content": contenu_tour_courant},
                            ],
                        )
                    except Exception as e2:
                        await message.reply(f"❌ {_erreur_ia_lisible(e2)}")
                        return
                else:
                    raise
            texte_reponse = reponse.choices[0].message.content or ""
        except Exception as e:
            await message.reply(f"❌ {_erreur_ia_lisible(e)}")
            return

        if not texte_reponse:
            texte_reponse = "(réponse vide, réessaie ta question)"

        historique.append({"role": "assistant", "content": texte_reponse})
        del historique[:-HISTORIQUE_MAX]
        await _maj_resume_si_necessaire(message.channel.id)

        for i in range(0, len(texte_reponse), 1900):
            await message.reply(texte_reponse[i:i + 1900])


@bot.tree.command(name="reset_chat", description="Efface la mémoire de conversation du chat IA dans ce salon")
async def reset_chat(interaction: discord.Interaction):
    historique_conversations.pop(interaction.channel_id, None)
    resume_conversations.pop(interaction.channel_id, None)
    await interaction.response.send_message("🧹 Mémoire de conversation effacée pour ce salon.", ephemeral=True)


@bot.tree.command(name="style", description="Change la personnalité du bot (accent/humeur) pour ce salon")
@app_commands.describe(personnalite="La personnalité à adopter pour le chat IA dans ce salon")
@app_commands.choices(personnalite=[
    app_commands.Choice(name="🌞 Marseillais", value="marseillais"),
    app_commands.Choice(name="🥖 Parigot", value="parigot"),
    app_commands.Choice(name="💼 Sérieux (SAV)", value="serieux"),
    app_commands.Choice(name="🔥 Vendeur hype", value="hype"),
    app_commands.Choice(name="🧊 Négociateur cash", value="cash"),
])
async def style(interaction: discord.Interaction, personnalite: app_commands.Choice[str]):
    style_actuel[interaction.channel_id] = personnalite.value
    # On efface la mémoire du salon pour éviter que l'IA mélange deux tons dans le même historique.
    historique_conversations.pop(interaction.channel_id, None)
    infos = STYLES[personnalite.value]
    await interaction.response.send_message(f"🎭 Personnalité changée : **{infos['nom']}**\n> {infos['intro']}")


# ============================================================
#  Galerie photo (flèches) pour les annonces comparables
# ============================================================

class GalerieView(discord.ui.View):
    """Permet de naviguer entre les annonces comparables avec des flèches, une photo à la fois."""

    MEDAILLES = ["🥇", "🥈", "🥉", "🏅", "🏅"]
    COULEURS = [
        discord.Color.gold(),
        discord.Color.light_grey(),
        discord.Color.dark_orange(),
        discord.Color.blurple(),
        discord.Color.blurple(),
    ]

    def __init__(self, items, embed_principal):
        super().__init__(timeout=600)
        self.items = items
        self.index = 0
        self.embed_principal = embed_principal
        self.message = None  # assigné après l'envoi, pour pouvoir désactiver les boutons à l'expiration
        self._maj_etat_boutons()

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass

    def _maj_etat_boutons(self):
        self.precedent.disabled = self.index == 0
        self.suivant.disabled = self.index >= len(self.items) - 1

    def _embed_item_courant(self) -> discord.Embed:
        i = self.items[self.index]
        titre_annonce = _champ(i, "title") or "Sans titre"
        url_annonce = _champ(i, "url", "")
        prix_annonce = _prix_de(i)
        favs = _favoris_de(i)
        photo_annonce = _photo_de(i)
        medaille = self.MEDAILLES[min(self.index, len(self.MEDAILLES) - 1)]
        couleur = self.COULEURS[min(self.index, len(self.COULEURS) - 1)]

        e = discord.Embed(
            title=f"{medaille} {titre_annonce[:90]}",
            url=url_annonce or None,
            description=f"**{prix_annonce} €**   •   ❤️ {favs} favoris\nAnnonce {self.index + 1}/{len(self.items)}",
            color=couleur,
        )
        if photo_annonce:
            e.set_image(url=photo_annonce)
        return e

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary)
    async def precedent(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index -= 1
        self._maj_etat_boutons()
        await interaction.response.edit_message(embeds=[self.embed_principal, self._embed_item_courant()], view=self)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary)
    async def suivant(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index += 1
        self._maj_etat_boutons()
        await interaction.response.edit_message(embeds=[self.embed_principal, self._embed_item_courant()], view=self)


# ============================================================
#  Fonction générique de recherche Vinted (réutilisée par /estimer et les alertes)
# ============================================================

async def _rechercher_vinted(requete: str, per_page: int = 50):
    """Cherche des annonces sur Vinted pour une requête donnée, avec tentatives multiples
    en cas de blocage temporaire. Retourne la liste brute d'items (dicts)."""
    url = f"https://www.vinted.fr/catalog?search_text={quote(requete)}&order=relevance"

    async def _rechercher():
        derniere_erreur = None
        for tentative in range(3):
            try:
                async with VintedClient(
                    persist_cookies=True,
                    cookies_dir=Path("/tmp/vinted_cookies"),
                ) as client:
                    return await client.search_items(url=url, per_page=per_page, raw_data=True)
            except (VintedAuthError, VintedRateLimitError) as e:
                derniere_erreur = e
                if tentative < 2:
                    await asyncio.sleep(4 * (tentative + 1))
        raise derniere_erreur

    return await asyncio.wait_for(_rechercher(), timeout=45)


_CACHE_RECHERCHE = {}  # {(requete_normalisee, per_page): (timestamp, items)}
CACHE_RECHERCHE_TTL = 180  # secondes — assez court pour rester frais, assez long pour éviter les doublons


async def _rechercher_vinted_cache(requete: str, per_page: int = 50):
    """Comme _rechercher_vinted, mais réutilise un résultat récent (< 3 min) pour la même requête au
    lieu de retaper Vinted : évite les appels redondants quand plusieurs personnes cherchent la même
    chose à peu de temps d'intervalle (recherche, estimation, alertes confondues)."""
    cle = (requete.strip().lower(), per_page)
    maintenant = asyncio.get_event_loop().time()
    entree = _CACHE_RECHERCHE.get(cle)
    if entree and (maintenant - entree[0]) < CACHE_RECHERCHE_TTL:
        return entree[1]

    items = await _rechercher_vinted(requete, per_page=per_page)
    _CACHE_RECHERCHE[cle] = (maintenant, items)

    # Purge légère du cache pour ne pas grossir indéfiniment (déclenchée seulement si ça devient gros).
    if len(_CACHE_RECHERCHE) > 200:
        expiration = maintenant - CACHE_RECHERCHE_TTL
        for c in [c for c, (t, _) in _CACHE_RECHERCHE.items() if t < expiration]:
            _CACHE_RECHERCHE.pop(c, None)

    return items


# ============================================================
#  Alertes de bonnes affaires Vinted (vérifiées en arrière-plan)
# ============================================================

ALERTES_FICHIER = Path("/tmp/alertes_vinted.json")
ALERTES_INTERVALLE_MINUTES = int(os.environ.get("ALERTES_INTERVALLE_MINUTES", "20"))
ALERTES_MAX_PAR_UTILISATEUR = 5
ALERTES_VUS_MAX = 200  # nb d'annonces déjà vues gardées par alerte, pour ne pas grossir indéfiniment


def _charger_alertes():
    if ALERTES_FICHIER.exists():
        try:
            return json.loads(ALERTES_FICHIER.read_text())
        except Exception as e:
            print(f"[alertes] échec de lecture du fichier, on repart de zéro : {e}")
    return []


async def _sauver_alertes():
    try:
        await asyncio.to_thread(ALERTES_FICHIER.write_text, json.dumps(alertes_actives))
    except Exception as e:
        print(f"[alertes] échec de sauvegarde : {e}")


alertes_actives = _charger_alertes()
_compteur_id_alerte = itertools.count(max([a["id"] for a in alertes_actives], default=0) + 1)


async def _creer_alerte(user_id: int, article: str, prix_max: Optional[float]):
    """Crée une alerte de prix pour un utilisateur. Retourne (ok, message) — ok=False si le quota
    d'alertes actives est atteint."""
    mes_alertes = [a for a in alertes_actives if a["user_id"] == user_id]
    if len(mes_alertes) >= ALERTES_MAX_PAR_UTILISATEUR:
        return False, (
            f"⚠️ Tu as déjà {ALERTES_MAX_PAR_UTILISATEUR} alertes actives (le maximum). "
            "Supprime-en une avec `/alerte_supprimer` avant d'en ajouter une nouvelle."
        )

    nouvelle = {
        "id": next(_compteur_id_alerte),
        "user_id": user_id,
        "article": article.strip(),
        "prix_max": prix_max,
        "vus": [],
    }
    alertes_actives.append(nouvelle)
    await _sauver_alertes()

    texte_prix = f" à moins de {prix_max:.2f} €" if prix_max else ""
    return True, (
        f"🔔 Alerte `#{nouvelle['id']}` créée pour **{nouvelle['article']}**{texte_prix} !\n"
        f"Je t'enverrai un message privé dès qu'une annonce correspondante apparaît "
        f"(vérification toutes les ~{ALERTES_INTERVALLE_MINUTES} minutes). "
        "Vérifie que tes MPs sont ouverts sur ce serveur pour bien recevoir l'alerte."
    )


@bot.tree.command(name="alerte_ajouter", description="Sois prévenu(e) par MP dès qu'une bonne affaire correspondante apparaît sur Vinted")
@app_commands.describe(
    article="Ce que tu cherches (ex: nike air force taille 42)",
    prix_max="Prix maximum en € (optionnel, sinon toute annonce correspondante déclenche l'alerte)",
)
async def alerte_ajouter(interaction: discord.Interaction, article: str, prix_max: Optional[float] = None):
    ok, message = await _creer_alerte(interaction.user.id, article, prix_max)
    await interaction.response.send_message(message, ephemeral=True)


@bot.tree.command(name="alerte_liste", description="Affiche tes alertes actives")
async def alerte_liste(interaction: discord.Interaction):
    mes_alertes = [a for a in alertes_actives if a["user_id"] == interaction.user.id]
    if not mes_alertes:
        await interaction.response.send_message(
            "Tu n'as aucune alerte active. Crée-en une avec `/alerte_ajouter`.", ephemeral=True
        )
        return

    lignes = []
    for a in mes_alertes:
        prix = f" (max {a['prix_max']:.2f} €)" if a.get("prix_max") else ""
        lignes.append(f"`#{a['id']}` — {a['article']}{prix}")

    embed = discord.Embed(
        title="🔔 Tes alertes actives",
        description="\n".join(lignes),
        color=discord.Color.blurple(),
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="alerte_supprimer", description="Supprime une de tes alertes")
@app_commands.describe(id="Le numéro de l'alerte (visible avec /alerte_liste)")
async def alerte_supprimer(interaction: discord.Interaction, id: int):
    global alertes_actives
    avant = len(alertes_actives)
    alertes_actives[:] = [a for a in alertes_actives if not (a["id"] == id and a["user_id"] == interaction.user.id)]
    await _sauver_alertes()

    if len(alertes_actives) < avant:
        await interaction.response.send_message(f"🗑️ Alerte `#{id}` supprimée.", ephemeral=True)
    else:
        await interaction.response.send_message(
            "Aucune alerte trouvée avec ce numéro (elle ne t'appartient peut-être pas).", ephemeral=True
        )


@tasks.loop(minutes=ALERTES_INTERVALLE_MINUTES)
async def verifier_alertes():
    if not alertes_actives:
        return

    # Regroupe les alertes par requête identique (insensible à la casse/espaces) : si 5 personnes
    # suivent "nike air force 42", on ne fait qu'UN SEUL appel Vinted pour les 5, au lieu de 5 appels
    # séquentiels — plus rapide, et beaucoup plus doux avec Vinted (moins de risque de blocage).
    groupes = {}
    for alerte in list(alertes_actives):
        cle = alerte["article"].strip().lower()
        groupes.setdefault(cle, []).append(alerte)

    for cle, alertes_du_groupe in groupes.items():
        requete = alertes_du_groupe[0]["article"]
        try:
            items = await _rechercher_vinted_cache(requete, per_page=20)
        except Exception as e:
            print(f"[alertes] échec recherche pour '{requete}' ({len(alertes_du_groupe)} alerte(s) concernée(s)) : {e}")
            await asyncio.sleep(5)
            continue

        for alerte in alertes_du_groupe:
            nouveaux = []
            for item in items or []:
                item_id = str(_champ(item, "id", _champ(item, "url", "")))
                if not item_id or item_id in alerte["vus"]:
                    continue
                prix = _prix_de(item)
                if alerte.get("prix_max") and (prix is None or prix > alerte["prix_max"]):
                    continue
                nouveaux.append(item)
                alerte["vus"].append(item_id)

            alerte["vus"] = alerte["vus"][-ALERTES_VUS_MAX:]

            if nouveaux:
                try:
                    utilisateur = await bot.fetch_user(alerte["user_id"])
                    for item in nouveaux[:3]:  # au plus 3 notifs par vérification, pour ne pas spammer
                        titre = _champ(item, "title") or "Sans titre"
                        prix = _prix_de(item)
                        url_item = _champ(item, "url", "")
                        photo_item = _photo_de(item)
                        embed = discord.Embed(
                            title=f"🔔 Bonne affaire trouvée : {titre[:100]}",
                            url=url_item or None,
                            description=f"**{prix} €**\nCorrespond à ton alerte : *{alerte['article']}*",
                            color=discord.Color.gold(),
                        )
                        if photo_item:
                            embed.set_image(url=photo_item)
                        await utilisateur.send(embed=embed)
                except discord.Forbidden:
                    print(f"[alertes] MPs fermés pour l'utilisateur {alerte['user_id']}, notification impossible.")
                except Exception as e:
                    print(f"[alertes] échec d'envoi de notification : {e}")

        await asyncio.sleep(3)  # petite pause entre chaque requête UNIQUE, pour ménager Vinted

    await _sauver_alertes()


@verifier_alertes.before_loop
async def avant_verifier_alertes():
    await bot.wait_until_ready()


# ============================================================
#  Suivi des ventes & tableau de bord personnel
# ============================================================

VENTES_FICHIER = Path("/tmp/ventes_utilisateurs.json")


def _charger_ventes():
    if VENTES_FICHIER.exists():
        try:
            return json.loads(VENTES_FICHIER.read_text())
        except Exception as e:
            print(f"[ventes] échec de lecture du fichier, on repart de zéro : {e}")
    return []


async def _sauver_ventes():
    try:
        await asyncio.to_thread(VENTES_FICHIER.write_text, json.dumps(ventes_enregistrees))
    except Exception as e:
        print(f"[ventes] échec de sauvegarde : {e}")


ventes_enregistrees = _charger_ventes()
_compteur_id_vente = itertools.count(max([v["id"] for v in ventes_enregistrees], default=0) + 1)


def _barre_texte(valeur, valeur_max, longueur=10):
    if valeur_max <= 0:
        rempli = 0
    else:
        rempli = min(longueur, round((valeur / valeur_max) * longueur))
    return "▰" * rempli + "▱" * (longueur - rempli)


def _embed_liste_ventes(interaction: discord.Interaction) -> Optional[discord.Embed]:
    mes_ventes = [v for v in ventes_enregistrees if v["user_id"] == interaction.user.id]
    if not mes_ventes:
        return None
    dernieres = mes_ventes[-15:][::-1]
    lignes = []
    for v in dernieres:
        prix_achat_txt = f" (achetée {v['prix_achat']:.2f} €)" if v.get("prix_achat") is not None else ""
        prix_annonce = v.get("prix_annonce")
        if prix_annonce is not None and prix_annonce != v["prix_vente"]:
            texte_prix = f"annoncée {prix_annonce:.2f} € → **{v['prix_vente']:.2f} €**"
        else:
            texte_prix = f"**{v['prix_vente']:.2f} €**"
        photo_txt = " 📷" if v.get("photo_url") else ""
        lignes.append(f"`#{v['id']}` — {v['article']} — {texte_prix}{prix_achat_txt}{photo_txt}")
    embed = discord.Embed(
        title="🧾 Tes dernières ventes",
        description="\n".join(lignes),
        color=discord.Color.blurple(),
    )
    if len(mes_ventes) > 15:
        embed.set_footer(text=f"15 plus récentes sur {len(mes_ventes)} au total")
    return embed


def _embed_bilan_ventes(interaction: discord.Interaction) -> Optional[discord.Embed]:
    mes_ventes = [v for v in ventes_enregistrees if v["user_id"] == interaction.user.id]
    if not mes_ventes:
        return None

    nb_ventes = len(mes_ventes)
    chiffre_affaires = sum(v["prix_vente"] for v in mes_ventes)
    ventes_avec_achat = [v for v in mes_ventes if v.get("prix_achat") is not None]
    benefice_total = sum(v["prix_vente"] - v["prix_achat"] for v in ventes_avec_achat)
    marge_moyenne = (
        statistics.mean(
            ((v["prix_vente"] - v["prix_achat"]) / v["prix_achat"] * 100)
            for v in ventes_avec_achat if v["prix_achat"] > 0
        )
        if ventes_avec_achat else None
    )

    meilleure_vente = max(mes_ventes, key=lambda v: v["prix_vente"])
    if ventes_avec_achat:
        meilleur_benefice = max(ventes_avec_achat, key=lambda v: v["prix_vente"] - v["prix_achat"])
    else:
        meilleur_benefice = None

    embed = discord.Embed(
        title=f"📊 Tableau de bord de {interaction.user.display_name}",
        color=discord.Color.from_rgb(255, 215, 0),
    )
    embed.set_thumbnail(url=interaction.user.display_avatar.url)

    premiere_date = min(datetime.fromisoformat(v["date"]) for v in mes_ventes)
    jours_actif = (discord.utils.utcnow() - premiere_date).days
    texte_depuis = f"Le {premiere_date.strftime('%d/%m/%Y')}" + (f" ({jours_actif} jour(s))" if jours_actif > 0 else " (aujourd'hui)")
    embed.add_field(name="🗓️ Vendeur depuis", value=texte_depuis, inline=False)

    embed.add_field(name="🧾 Ventes totales", value=str(nb_ventes), inline=True)
    embed.add_field(name="💶 Chiffre d'affaires", value=f"{chiffre_affaires:.2f} €", inline=True)
    if ventes_avec_achat:
        embed.add_field(name="📈 Bénéfice cumulé", value=f"{benefice_total:+.2f} €", inline=True)
        embed.add_field(name="📐 Marge moyenne", value=f"{marge_moyenne:.0f} %", inline=True)
    else:
        embed.add_field(
            name="📈 Bénéfice cumulé",
            value="Renseigne le prix d'achat sur tes prochaines ventes pour l'activer.",
            inline=False,
        )

    ventes_avec_delai = [v for v in mes_ventes if v.get("jours_en_ligne") is not None]
    if ventes_avec_delai:
        delai_moyen = statistics.mean(v["jours_en_ligne"] for v in ventes_avec_delai)
        embed.add_field(name="⏱️ Temps de vente moyen", value=f"{_formater_duree(delai_moyen)} en ligne avant la vente", inline=True)

    embed.add_field(
        name="🏆 Meilleure vente (prix)",
        value=f"{meilleure_vente['article']} — **{meilleure_vente['prix_vente']:.2f} €**",
        inline=False,
    )
    if meilleur_benefice:
        gain = meilleur_benefice["prix_vente"] - meilleur_benefice["prix_achat"]
        embed.add_field(
            name="💎 Meilleure vente (bénéfice)",
            value=f"{meilleur_benefice['article']} — **+{gain:.2f} €**",
            inline=False,
        )

    cinq_dernieres = mes_ventes[-5:]
    prix_max_recent = max(v["prix_vente"] for v in cinq_dernieres)
    lignes_graphique = [
        f"{_barre_texte(v['prix_vente'], prix_max_recent)}  {v['prix_vente']:.0f}€ — {v['article'][:25]}"
        for v in cinq_dernieres
    ]
    embed.add_field(name="📉 5 dernières ventes", value="\n".join(lignes_graphique), inline=False)
    embed.set_footer(text="Historique stocké sur le serveur du bot • bouton ➕ pour continuer le suivi")
    return embed


def _parser_duree_en_jours(texte: str) -> Optional[float]:
    """Extrait un nombre de jours d'un texte de durée écrit librement (ex: '5', '5j', '2 semaines',
    '1 semaine 3 jours', '10h', 'environ 3 jours'...). Additionne tous les segments nombre+unité
    trouvés, où qu'ils soient dans le texte — jamais de rejet strict, l'utilisateur doit pouvoir
    écrire comme il veut. Retourne None seulement si vraiment aucun nombre n'a pu être trouvé."""
    if not texte or not texte.strip():
        return None
    t = texte.strip().lower().replace(",", ".")
    total = 0.0
    trouve = False
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(heures?|h\b|semaines?|sem\.?|mois|jours?|jrs?|j\b)?", t):
        valeur = float(match.group(1))
        unite = (match.group(2) or "").strip().rstrip(".")
        trouve = True
        if unite.startswith("heure") or unite == "h":
            total += valeur / 24
        elif unite.startswith("sem"):
            total += valeur * 7
        elif unite == "mois":
            total += valeur * 30
        else:
            total += valeur  # pas d'unité, ou 'j'/'jour(s)'/'jrs' → traité comme des jours
    return total if trouve else None


def _formater_duree(jours: float) -> str:
    if jours < 1:
        return f"{round(jours * 24)} heure(s)"
    if jours == int(jours):
        return f"{int(jours)} jour(s)"
    return f"{jours:.1f} jour(s)"


class VenteAjouterModal(discord.ui.Modal, title="➕ Nouvelle vente"):
    article = discord.ui.TextInput(label="Article vendu", placeholder="ex: Pull Nike taille M", required=True, max_length=200)
    prix_annonce = discord.ui.TextInput(
        label="Prix affiché sur l'annonce en € (optionnel)",
        placeholder="ex: 30 — laisse vide si identique au prix vendu",
        required=False,
        max_length=10,
    )
    prix_vente = discord.ui.TextInput(label="Prix auquel elle a été vendue en €", placeholder="ex: 25", required=True, max_length=10)
    prix_achat = discord.ui.TextInput(
        label="Prix que tu l'avais payé en € (optionnel)",
        placeholder="ex: 10 — laisse vide si tu ne sais pas",
        required=False,
        max_length=10,
    )
    temps_en_ligne = discord.ui.TextInput(
        label="Temps en ligne avant la vente (optionnel)",
        placeholder="ex: 5j, 2 semaines, 10h — laisse vide si tu ne sais pas",
        required=False,
        max_length=20,
    )

    def __init__(self, photo: Optional[discord.Attachment] = None, commande_liee: Optional[dict] = None):
        super().__init__()
        self.photo = photo
        self.commande_liee = commande_liee
        if commande_liee:
            self.article.default = commande_liee["article"][:200]
            if commande_liee.get("prix_prevu") is not None:
                self.prix_achat.default = str(commande_liee["prix_prevu"])

    async def on_submit(self, interaction: discord.Interaction):
        try:
            prix_vente_val = float(self.prix_vente.value.strip().replace(",", "."))
        except ValueError:
            await interaction.response.send_message("⚠️ Le prix de vente doit être un nombre (ex: 25 ou 24.90).", ephemeral=True)
            return

        prix_achat_val = None
        if self.prix_achat.value and self.prix_achat.value.strip():
            try:
                prix_achat_val = float(self.prix_achat.value.strip().replace(",", "."))
            except ValueError:
                await interaction.response.send_message("⚠️ Le prix d'achat doit être un nombre (ex: 10 ou 9.90).", ephemeral=True)
                return

        prix_annonce_val = None
        if self.prix_annonce.value and self.prix_annonce.value.strip():
            try:
                prix_annonce_val = float(self.prix_annonce.value.strip().replace(",", "."))
            except ValueError:
                await interaction.response.send_message("⚠️ Le prix de l'annonce doit être un nombre (ex: 30 ou 29.90).", ephemeral=True)
                return

        temps_en_ligne_texte = self.temps_en_ligne.value.strip() if self.temps_en_ligne.value else ""
        jours_en_ligne_val = _parser_duree_en_jours(temps_en_ligne_texte) if temps_en_ligne_texte else None
        # Si vraiment aucun nombre n'a pu être extrait (ex: "je sais pas"), on n'utilise juste pas cette
        # info dans les moyennes du bilan, mais on n'empêche JAMAIS d'enregistrer la vente pour ça.

        date_vente = discord.utils.utcnow()
        date_mise_en_ligne = (date_vente - timedelta(days=jours_en_ligne_val)) if jours_en_ligne_val is not None else None

        nouvelle = {
            "id": next(_compteur_id_vente),
            "user_id": interaction.user.id,
            "article": self.article.value.strip()[:200],
            "prix_annonce": prix_annonce_val,
            "prix_vente": prix_vente_val,
            "prix_achat": prix_achat_val,
            "jours_en_ligne": jours_en_ligne_val,
            "temps_en_ligne_texte": temps_en_ligne_texte or None,
            "date": date_vente.isoformat(),
            "date_mise_en_ligne": date_mise_en_ligne.isoformat() if date_mise_en_ligne else None,
            "photo_url": self.photo.url if self.photo else None,
            "commande_id": self.commande_liee["id"] if self.commande_liee else None,
        }
        ventes_enregistrees.append(nouvelle)
        await _sauver_ventes()

        if self.commande_liee:
            for c in commandes_prevues:
                if c["id"] == self.commande_liee["id"]:
                    c["statut"] = "vendue"
                    c["vente_id"] = nouvelle["id"]
                    break
            await _sauver_commandes()

        if prix_annonce_val is not None and prix_annonce_val != prix_vente_val:
            texte_prix = f"annoncée à {prix_annonce_val:.2f} € → vendue **{prix_vente_val:.2f} €**"
        else:
            texte_prix = f"vendue **{prix_vente_val:.2f} €**"
        description = f"**{nouvelle['article']}** — {texte_prix}"
        if prix_achat_val is not None:
            benefice = round(prix_vente_val - prix_achat_val, 2)
            description += f" (achetée {prix_achat_val:.2f} € → {'bénéfice' if benefice >= 0 else 'perte'} de {abs(benefice):.2f} €)"
        if temps_en_ligne_texte:
            description += f"\n🕐 En ligne **{temps_en_ligne_texte}** avant la vente."
        if self.commande_liee:
            description += f"\n🔗 Liée à la commande `#{self.commande_liee['id']}`, désormais marquée comme vendue."
        description += "\nRelance `/suivi` puis 📊 Mon bilan pour voir ton bilan complet."

        embed = discord.Embed(title=f"✅ Vente #{nouvelle['id']} enregistrée", description=description, color=discord.Color.green())
        if self.photo:
            embed.set_thumbnail(url=self.photo.url)
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        # Pas de photo jointe à la commande /vente elle-même (les modals Discord ne peuvent pas en
        # contenir) : on propose d'en envoyer une juste après, pour l'associer à cette vente précise —
        # pratique pour enchaîner plusieurs ventes avec une photo différente à chaque fois.
        embed.set_footer(text="📷 Envoie une photo ici dans les 90 secondes pour l'associer à cette vente (facultatif).")
        await interaction.response.send_message(embed=embed, ephemeral=True)

        def _verif_photo(m: discord.Message) -> bool:
            return (
                m.author.id == interaction.user.id
                and m.channel.id == interaction.channel.id
                and bool(_images_jointes(m))
            )

        try:
            message_photo = await bot.wait_for("message", check=_verif_photo, timeout=90)
        except asyncio.TimeoutError:
            return

        photo_recue = _images_jointes(message_photo)[0]
        nouvelle["photo_url"] = photo_recue.url
        await _sauver_ventes()
        await interaction.followup.send(f"📷 Photo associée à la vente `#{nouvelle['id']}` !", ephemeral=True)


class VenteSupprimerModal(discord.ui.Modal, title="🗑️ Supprimer une vente"):
    id_vente = discord.ui.TextInput(
        label="Numéro de la vente (# visible via 🧾)",
        placeholder="ex: 3",
        required=True,
        max_length=10,
    )

    async def on_submit(self, interaction: discord.Interaction):
        try:
            id_val = int(self.id_vente.value.strip().lstrip("#"))
        except ValueError:
            await interaction.response.send_message("⚠️ Le numéro doit être un nombre entier (ex: 3).", ephemeral=True)
            return

        avant = len(ventes_enregistrees)
        ventes_enregistrees[:] = [
            v for v in ventes_enregistrees if not (v["id"] == id_val and v["user_id"] == interaction.user.id)
        ]
        await _sauver_ventes()

        if len(ventes_enregistrees) < avant:
            await interaction.response.send_message(f"🗑️ Vente `#{id_val}` supprimée.", ephemeral=True)
        else:
            await interaction.response.send_message(
                "Aucune vente trouvée avec ce numéro (elle ne t'appartient peut-être pas).", ephemeral=True
            )


def _mes_commandes_passees(user_id: int) -> list:
    """Commandes achetées (statut 'passee') mais pas encore mises en ligne, les plus récentes en
    premier — ce sont les candidates pour être marquées 'en ligne'."""
    mes_commandes = [c for c in commandes_prevues if c["user_id"] == user_id and c["statut"] == "passee"]
    return list(reversed(mes_commandes))[:25]  # limite Discord : 25 options max dans un menu déroulant


def _mes_commandes_en_ligne(user_id: int) -> list:
    """Commandes déjà mises en ligne (statut 'en_ligne') mais pas encore vendues, les plus
    récentes en premier — ce sont les candidates pour être liées à une nouvelle vente."""
    mes_commandes = [c for c in commandes_prevues if c["user_id"] == user_id and c["statut"] == "en_ligne"]
    return list(reversed(mes_commandes))[:25]


class CommandeEnLigneModal(discord.ui.Modal, title="📤 Mettre en ligne"):
    prix_annonce = discord.ui.TextInput(
        label="Prix affiché sur l'annonce en € (optionnel)",
        placeholder="ex: 30 — laisse vide si tu ne sais pas encore",
        required=False,
        max_length=10,
    )

    def __init__(self, commande: dict):
        super().__init__()
        self.commande = commande

    async def on_submit(self, interaction: discord.Interaction):
        prix_annonce_val = None
        if self.prix_annonce.value and self.prix_annonce.value.strip():
            try:
                prix_annonce_val = float(self.prix_annonce.value.strip().replace(",", "."))
            except ValueError:
                await interaction.response.send_message("⚠️ Le prix doit être un nombre (ex: 30 ou 29.90).", ephemeral=True)
                return

        for c in commandes_prevues:
            if c["id"] == self.commande["id"] and c["user_id"] == interaction.user.id:
                c["statut"] = "en_ligne"
                if prix_annonce_val is not None:
                    c["prix_prevu"] = prix_annonce_val
                break
        await _sauver_commandes()

        await interaction.response.send_message(
            f"📤 **{self.commande['article']}** marqué comme en ligne !\n"
            "Une fois vendu, choisis 💰 **Marquer vendu** dans `/suivi` pour finaliser la vente.",
            ephemeral=True,
        )


class CommandeSelectPourEnLigne(discord.ui.Select):
    def __init__(self, commandes: list):
        options = [
            discord.SelectOption(
                label=c["article"][:100],
                description=f"Commande #{c['id']}"[:100],
                value=str(c["id"]),
            )
            for c in commandes
        ]
        super().__init__(placeholder="Quel article viens-tu de mettre en ligne ?", options=options)

    async def callback(self, interaction: discord.Interaction):
        id_choisi = int(self.values[0])
        commande = next(
            (c for c in commandes_prevues if c["id"] == id_choisi and c["user_id"] == interaction.user.id),
            None,
        )
        if commande is None or commande["statut"] != "passee":
            await interaction.response.send_message(
                "⚠️ Cette commande n'est plus disponible (déjà mise en ligne entre-temps ?).",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(CommandeEnLigneModal(commande))


class CommandeSelectPourEnLigneView(discord.ui.View):
    def __init__(self, commandes: list):
        super().__init__(timeout=180)
        self.add_item(CommandeSelectPourEnLigne(commandes))


class CommandeSelectPourVente(discord.ui.Select):
    def __init__(self, commandes: list, photo: Optional[discord.Attachment] = None):
        self.photo = photo
        options = []
        for c in commandes:
            prix_txt = f" — ~{c['prix_prevu']:.2f} €" if c.get("prix_prevu") is not None else ""
            options.append(discord.SelectOption(
                label=c["article"][:100],
                description=f"Commande #{c['id']}{prix_txt}"[:100],
                value=str(c["id"]),
            ))
        super().__init__(placeholder="Quel article viens-tu de vendre ?", options=options)

    async def callback(self, interaction: discord.Interaction):
        id_choisi = int(self.values[0])
        commande = next(
            (c for c in commandes_prevues if c["id"] == id_choisi and c["user_id"] == interaction.user.id),
            None,
        )
        if commande is None or commande["statut"] != "en_ligne":
            await interaction.response.send_message(
                "⚠️ Cet article n'est plus disponible (déjà lié à une vente entre-temps ?).",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(VenteAjouterModal(photo=self.photo, commande_liee=commande))


class CommandeSelectPourVenteView(discord.ui.View):
    def __init__(self, commandes: list, photo: Optional[discord.Attachment] = None):
        super().__init__(timeout=180)
        self.add_item(CommandeSelectPourVente(commandes, photo=photo))


class SuiviSelect(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="Ajouter une commande", value="commande_ajouter", emoji="🛒",
                                  description="Note un article que tu comptes acheter pour revendre"),
            discord.SelectOption(label="Commandes à passer", value="commande_liste", emoji="📋",
                                  description="Ce que tu comptes encore acheter"),
            discord.SelectOption(label="Marquer une commande passée", value="commande_marquer", emoji="✅",
                                  description="Une fois l'achat fait"),
            discord.SelectOption(label="Supprimer une commande", value="commande_supprimer", emoji="🗑️",
                                  description="Corrige une erreur de saisie"),
            discord.SelectOption(label="Ajouter un article en ligne", value="commande_en_ligne", emoji="📤",
                                  description="Marque une commande achetée comme mise en vente"),
            discord.SelectOption(label="Marquer vendu", value="vente_depuis_commande", emoji="💰",
                                  description="Finalise la vente d'un article en ligne (pré-rempli)"),
            discord.SelectOption(label="Mes ventes", value="vente_liste", emoji="🧾",
                                  description="Tes 15 dernières ventes"),
            discord.SelectOption(label="Supprimer une vente", value="vente_supprimer", emoji="🗑️",
                                  description="Corrige une erreur de saisie"),
            discord.SelectOption(label="Mon bilan", value="vente_bilan", emoji="📊",
                                  description="Chiffre d'affaires, bénéfice, meilleures ventes"),
        ]
        super().__init__(placeholder="Que veux-tu faire ?", options=options, custom_id="suivi_select")

    async def callback(self, interaction: discord.Interaction):
        choix = self.values[0]
        photo = getattr(self.view, "photo", None)

        if choix == "commande_ajouter":
            await interaction.response.send_modal(CommandeAjouterModal())

        elif choix == "commande_liste":
            embed = _embed_liste_commandes(interaction, "a_passer")
            if embed is None:
                await interaction.response.send_message(
                    "Aucune commande en attente. Choisis 🛒 Ajouter une commande pour en noter une.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(embed=embed, ephemeral=True)

        elif choix == "commande_marquer":
            await interaction.response.send_modal(CommandeMarquerPasseeModal())

        elif choix == "commande_supprimer":
            await interaction.response.send_modal(CommandeSupprimerModal())

        elif choix == "commande_en_ligne":
            commandes = _mes_commandes_passees(interaction.user.id)
            if not commandes:
                await interaction.response.send_message(
                    "Tu n'as aucune commande achetée en attente de mise en ligne. Marque d'abord une "
                    "commande comme passée (✅), elle apparaîtra ici ensuite.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(
                "Quel article viens-tu de mettre en ligne ? 👇",
                view=CommandeSelectPourEnLigneView(commandes),
                ephemeral=True,
            )

        elif choix == "vente_depuis_commande":
            commandes = _mes_commandes_en_ligne(interaction.user.id)
            if not commandes:
                await interaction.response.send_message(
                    "Tu n'as aucun article en ligne en attente de vente. Choisis d'abord 📤 Ajouter un "
                    "article en ligne, il apparaîtra ici une fois vendu.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(
                "Quel article viens-tu de vendre ? 👇 (le formulaire de vente sera pré-rempli avec "
                "l'article et le prix d'achat prévu) :",
                view=CommandeSelectPourVenteView(commandes, photo=photo),
                ephemeral=True,
            )

        elif choix == "vente_liste":
            embed = _embed_liste_ventes(interaction)
            if embed is None:
                await interaction.response.send_message(
                    "Aucune vente enregistrée. Choisis 💰 Marquer vendu une fois un article en ligne vendu.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(embed=embed, ephemeral=True)

        elif choix == "vente_supprimer":
            await interaction.response.send_modal(VenteSupprimerModal())

        elif choix == "vente_bilan":
            embed = _embed_bilan_ventes(interaction)
            if embed is None:
                await interaction.response.send_message(
                    "Tu n'as encore aucune vente enregistrée. Ajoute ta première vente pour construire "
                    "ton tableau de bord.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(embed=embed, ephemeral=True)


class SuiviView(discord.ui.View):
    def __init__(self, photo: Optional[discord.Attachment] = None):
        super().__init__(timeout=None)
        self.photo = photo
        self.add_item(SuiviSelect())


@bot.tree.command(name="suivi", description="Suivi complet achat/revente : commandes, mise en ligne et ventes, tout en un seul endroit")
@app_commands.describe(photo="Photo de l'article vendu (optionnel, jointe si tu choisis Marquer vendu)")
async def suivi(interaction: discord.Interaction, photo: Optional[discord.Attachment] = None):
    embed = discord.Embed(
        title="🔁 Suivi achat/revente",
        description=(
            "Tout le cycle en un seul endroit, du repérage à la vente.\n\n"
            "Choisis une action dans le menu déroulant ci-dessous 👇\n\n"
            "**Le cycle complet**\n"
            "🛒 Ajouter une commande → ✅ Marquer passée (achetée) → 📤 Ajouter un article en ligne "
            "(mis en vente) → 💰 Marquer vendu (finalise la vente, pré-rempli)\n\n"
            "**Autres actions**\n"
            "📋 À passer · 🗑️ Supprimer une commande · 🧾 Mes ventes · 🗑️ Supprimer une vente · 📊 Mon bilan"
        ),
        color=discord.Color.from_rgb(255, 190, 60),
    )
    if photo:
        embed.set_thumbnail(url=photo.url)
    embed.set_footer(text="Chaque personne ne voit que ses propres commandes et ventes, en privé.")
    await interaction.response.send_message(embed=embed, view=SuiviView(photo=photo))



# ============================================================
#  Logique commune : lance la recherche et construit le résultat
# ============================================================

async def _lancer_estimation(
    interaction: discord.Interaction,
    article: str,
    prix_achat: Optional[float],
    marge_cible: Optional[float],
    photo: Optional[discord.Attachment],
):
    requete = article.strip()
    print(f"[estimer] requête envoyée à Vinted : {requete!r}")

    try:
        items = await _rechercher_vinted_cache(requete, per_page=50)
        print(f"[estimer] nb_items={len(items) if items else 0}")

    except asyncio.TimeoutError:
        await interaction.followup.send("⏱️ Vinted met trop de temps à répondre. Réessaie dans quelques minutes.", ephemeral=True)
        return
    except VintedAuthError:
        await interaction.followup.send(
            "🚫 Vinted a temporairement bloqué la connexion (ça arrive régulièrement avec les hébergeurs gratuits). "
            "Ce n'est pas systématique — réessaie dans quelques minutes, ça passe souvent au 2e ou 3e essai.",
            ephemeral=True,
        )
        return
    except VintedRateLimitError:
        await interaction.followup.send(
            "🚫 Vinted a temporairement limité les requêtes (trop de recherches d'un coup). "
            "Réessaie dans quelques minutes.",
            ephemeral=True,
        )
        return
    except (VintedNetworkError, VintedAPIError, VintedError) as e:
        await interaction.followup.send(f"❌ Erreur Vinted : `{e}`", ephemeral=True)
        return
    except Exception as e:
        await interaction.followup.send(f"❌ Erreur inattendue pendant la recherche : `{e}`", ephemeral=True)
        return

    if not items:
        await interaction.followup.send("Aucune annonce comparable trouvée. Essaie une description plus générale.", ephemeral=True)
        return

    # --- Prix ---
    prix_bruts = [p for p in (_prix_de(i) for i in items) if p is not None]
    if not prix_bruts:
        await interaction.followup.send("Impossible de récupérer des prix exploitables sur ces annonces.", ephemeral=True)
        return

    prix_filtres, nb_exclus = _filtrer_valeurs_extremes(prix_bruts)

    prix_moyen = statistics.mean(prix_filtres)
    prix_median = statistics.median(prix_filtres)
    prix_min = min(prix_filtres)
    prix_max = max(prix_filtres)

    favoris = [_favoris_de(i) for i in items]
    demande_moyenne = statistics.mean(favoris) if favoris else 0

    # Prix conseillé : basé sur le prix médian du marché, avec une marge adaptée
    # au prix d'achat si fourni (plus élevée pour les articles bon marché).
    marge_avertissement = None
    pourcentage_marge = None
    if prix_achat:
        pourcentage_marge = marge_cible if marge_cible is not None else _marge_cible_par_defaut(prix_achat)
        prix_minimum_rentable = round(prix_achat * (1 + pourcentage_marge / 100), 2)
        prix_conseille = max(prix_median, prix_minimum_rentable)
        if prix_conseille > prix_max:
            prix_conseille = prix_max
            if prix_conseille < prix_minimum_rentable:
                marge_avertissement = (
                    f"⚠️ Le marché ne permet pas la marge visée ({pourcentage_marge:.0f}%) sur cet article "
                    f"(prix max observé : {prix_max:.2f} €)."
                )
    else:
        prix_conseille = max(prix_median, prix_min)

    # --- Embed principal (statistiques) ---
    barre = "▰" * min(10, round(demande_moyenne / 5)) + "▱" * (10 - min(10, round(demande_moyenne / 5)))

    embed_principal = discord.Embed(
        title="📊 Estimation de prix",
        color=discord.Color.from_rgb(30, 200, 120),
    )
    embed_principal.timestamp = discord.utils.utcnow()
    embed_principal.add_field(name="🛍️ Article", value=f"**{article}**", inline=False)
    embed_principal.add_field(name="💶 Prix moyen", value=f"{prix_moyen:.2f} €", inline=True)
    embed_principal.add_field(name="↔️ Fourchette", value=f"{prix_min:.2f} € – {prix_max:.2f} €", inline=True)
    embed_principal.add_field(
        name="❤️ Demande",
        value=f"{barre}\n{demande_moyenne:.0f} favoris en moyenne",
        inline=False,
    )
    embed_principal.add_field(
        name="💡 Prix conseillé",
        value=f"## {prix_conseille:.2f} €",
        inline=False,
    )
    if prix_achat:
        benefice = round(prix_conseille - prix_achat, 2)
        embed_principal.add_field(
            name="📈 Bénéfice estimé",
            value=(
                f"{'+' if benefice >= 0 else ''}{benefice:.2f} € "
                f"(acheté {prix_achat:.2f} € • marge visée {pourcentage_marge:.0f}%)"
            ),
            inline=False,
        )
    if marge_avertissement:
        embed_principal.add_field(name="⚠️ Attention", value=marge_avertissement, inline=False)
    sous_texte = f"Basé sur {len(items)} annonces comparables"
    if nb_exclus:
        sous_texte += f" ({nb_exclus} valeur(s) extrême(s) écartée(s) du calcul)"
    embed_principal.set_footer(text=f"{sous_texte} • Données publiques Vinted, à titre indicatif")
    if photo:
        embed_principal.set_thumbnail(url=photo.url)

    # --- Jusqu'à 5 annonces comparables les plus "demandées", à parcourir avec les flèches ---
    top = sorted(items, key=_favoris_de, reverse=True)[:5]

    if top:
        vue = GalerieView(top, embed_principal)
        vue.message = await interaction.followup.send(
            embeds=[embed_principal, vue._embed_item_courant()], view=vue, ephemeral=True
        )
    else:
        await interaction.followup.send(embed=embed_principal, ephemeral=True)


# ============================================================
#  Formulaire (modal) déclenché par le bouton
# ============================================================

class EstimerModal(discord.ui.Modal, title="🔍 Nouvelle estimation Vinted"):
    article = discord.ui.TextInput(
        label="Article à estimer",
        placeholder="ex: Nike Air Force running, taille 42, bon état",
        required=True,
        max_length=200,
    )
    prix_achat = discord.ui.TextInput(
        label="Prix d'achat en € (optionnel)",
        placeholder="ex: 10",
        required=False,
        max_length=10,
    )
    marge_cible = discord.ui.TextInput(
        label="Marge visée en % (optionnel)",
        placeholder="Laisse vide pour un calcul automatique",
        required=False,
        max_length=10,
    )

    def __init__(self, photo: Optional[discord.Attachment] = None):
        super().__init__()
        self.photo = photo

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)

        prix_achat_valeur = None
        if self.prix_achat.value:
            try:
                prix_achat_valeur = float(self.prix_achat.value.replace(",", "."))
            except ValueError:
                await interaction.followup.send(
                    f"⚠️ Le prix d'achat `{self.prix_achat.value}` n'est pas un nombre valide, "
                    "il a été ignoré pour cette estimation.",
                    ephemeral=True,
                )

        marge_cible_valeur = None
        if self.marge_cible.value:
            try:
                marge_cible_valeur = float(self.marge_cible.value.replace(",", "."))
            except ValueError:
                await interaction.followup.send(
                    f"⚠️ La marge `{self.marge_cible.value}` n'est pas un nombre valide, "
                    "elle a été ignorée (calcul automatique utilisé).",
                    ephemeral=True,
                )

        await _lancer_estimation(
            interaction,
            article=self.article.value,
            prix_achat=prix_achat_valeur,
            marge_cible=marge_cible_valeur,
            photo=self.photo,
        )


class EstimerIntroView(discord.ui.View):
    def __init__(self, photo: Optional[discord.Attachment] = None):
        super().__init__(timeout=None)
        self.photo = photo

    @discord.ui.button(label="Lancer une estimation", style=discord.ButtonStyle.success, emoji="🔍", custom_id="estimer_lancer_bouton")
    async def lancer(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(EstimerModal(photo=self.photo))


# ============================================================
#  Commande /estimer — affiche l'explication + le bouton
# ============================================================

@bot.tree.command(name="estimer", description="Estime le prix de vente d'un article à partir d'annonces Vinted comparables")
@app_commands.describe(
    photo="Photo de l'article à vendre (optionnel, juste pour l'affichage)"
)
async def estimer(interaction: discord.Interaction, photo: Optional[discord.Attachment] = None):
    embed = discord.Embed(
        title="🔍 Estimateur de prix Vinted",
        description=(
            "Ce petit outil t'aide à fixer le meilleur prix de vente pour un article.\n\n"
            "**Comment ça marche ?**\n"
            "1️⃣ Clique sur le bouton ci-dessous\n"
            "2️⃣ Décris l'article (marque, type, taille, état)\n"
            "3️⃣ Indique ce que tu l'as payé (optionnel, pour viser un bénéfice)\n"
            "4️⃣ Le bot analyse des dizaines d'annonces similaires sur Vinted et te propose "
            "le prix le plus adapté, avec les annonces comparables les plus populaires."
        ),
        color=discord.Color.from_rgb(88, 101, 242),
    )
    if photo:
        embed.set_thumbnail(url=photo.url)
    embed.set_footer(text="Données publiques Vinted, à titre indicatif.")

    view = EstimerIntroView(photo=photo)
    await interaction.response.send_message(embed=embed, view=view)


# ============================================================
#  Générateur de description d'annonce (/description)
# ============================================================

def _prompt_ton_annonce(ton_key: str, channel_id) -> str:
    if ton_key == "personnalite":
        style_key = _style_du_salon(channel_id)
        return (
            f"Adopte le style de personnalité suivant, mais garde en tête qu'il s'agit de rédiger une "
            f"annonce de vente (pas une conversation) : {STYLES[style_key]['prompt']}"
        )
    return TONS_ANNONCE.get(ton_key, TONS_ANNONCE["accrocheur"])


def _construire_instruction_annonce(ton_key: str, channel_id, avec_photo: bool) -> str:
    ton_description = _prompt_ton_annonce(ton_key, channel_id)
    note_photo = (
        "Une photo de l'article est fournie : appuie-toi dessus pour préciser couleur, matière ou état "
        "réellement visibles. Ne mentionne rien qui ne soit pas visible sur la photo ou fourni en texte.\n\n"
        if avec_photo else ""
    )
    return (
        "Tu es un copywriter expert des annonces de vente d'articles d'occasion sur Vinted, en France, "
        "qui s'inspire de ce que font les vendeurs les plus performants de la plateforme. Un vendeur te "
        "donne des mots-clés (et parfois des infos complémentaires ou une photo) et tu rédiges une "
        "annonce naturelle et vendeuse, comme si un vrai particulier l'avait écrite lui-même — jamais un "
        "texte marketing générique ou creux.\n\n"
        f"{note_photo}"
        "Structure OBLIGATOIRE de la description (c'est le format des meilleures annonces Vinted, avec "
        "des sauts de ligne réels \\n dans le champ JSON) :\n"
        "1. Une ligne d'accroche courte et percutante (une seule phrase), qui donne l'info la plus vendeuse "
        "en premier (état neuf, pièce rare, marque recherchée, bon plan). Les meilleurs vendeurs ne "
        "présentent jamais l'article, ils balancent directement ce qui fait sa valeur.\n"
        "2. Une ligne vide.\n"
        "3. Une liste de 3 à 5 lignes, CHACUNE commençant par un emoji pertinent et différent des autres "
        "(par exemple 📏 pour la taille/les mesures, 🎨 pour la couleur, 🧵 pour la matière, ✨ pour l'état, "
        "📦 pour l'envoi, 👟/👕 selon le type d'article...), une seule information concrète par ligne, sans "
        "jamais répéter une info déjà donnée dans l'accroche.\n"
        "4. Une ligne vide.\n"
        "5. Une phrase de clôture courte, naturelle et variée (question spécifique à l'article, ou "
        "mention d'un détail qui donne envie) — jamais une formule creuse ou un appel à l'action générique.\n\n"
        "Règles d'écriture (important, à respecter systématiquement) :\n"
        "- Majuscule en début de chaque phrase et de chaque ligne de la liste, ponctuation correcte "
        "partout. Un texte qui commence en minuscule ou sans ponctuation a l'air négligé, c'est interdit.\n"
        "- Chaque ligne de la liste doit sonner comme une remarque naturelle qu'un vrai vendeur ferait à "
        "l'oral, jamais comme une fiche technique. Bannis les mots creux de fiche produit pris seuls : "
        "'authentique', 'optimal', 'impeccable', 'de qualité' — s'ils apparaissent, ajoute une nuance "
        "concrète et personnelle à côté ('semelle épaisse, ça encaisse bien sur route' plutôt que "
        "'semelle avec amorti optimal').\n"
        "- Jamais un adjectif vague sans élément concret derrière (préfère une sensation ou un usage "
        "réel : 'tissu épais qui tient chaud l'hiver' plutôt que 'matière de qualité').\n\n"
        "Format de réponse :\n"
        "- Réponds STRICTEMENT avec un objet JSON valide, sans texte autour, sans balises markdown, sans "
        "clé supplémentaire.\n"
        '- Format exact : {"titre": "...", "description": "..."}\n'
        "- Titre : maximum 60 caractères, avec les infos clés (marque, type, taille si connues) et si "
        "possible un mot qui capte l'attention (ex: 'comme neuf', 'rare', 'édition limitée' — seulement si "
        "c'est vrai et pertinent), avec une majuscule en début de titre, sans majuscules excessives ni "
        "emoji dans le titre lui-même.\n"
        "- Description : entre 300 et 600 caractères au total, structurée en accroche + liste à puces + "
        "clôture comme décrit ci-dessus (jamais une liste de mots-clés recopiés tels quels, jamais de "
        "tournure de brochure publicitaire).\n\n"
        "Consignes anti-répétition (important, pour éviter un effet 'texte générique/robot') :\n"
        "- Ne mentionne CHAQUE information (marque, taille, état, matière...) qu'UNE SEULE fois dans tout "
        "le texte, jamais deux fois même reformulée différemment.\n"
        "- N'ouvre JAMAIS par 'Je mets en vente', 'Je vends', 'À vendre' ou toute variante — ce sont les "
        "pires accroches, elles noient l'info utile dans du remplissage. Démarre directement par ce qui "
        "vend : l'état, la marque, une caractéristique rare, ou une mise en situation concrète.\n"
        "- Bannis toutes les tournures creuses et trop vues : 'Parfait pour...', 'Idéal pour...', "
        "'N'attendez plus', 'Une vraie trouvaille', 'Ne manquez pas cette occasion', 'Le confort typique "
        "de la marque', 'N'hésitez pas à me poser vos/toute question(s)', 'Livraison rapide et soignée' "
        "— remplace-les par des détails réels et spécifiques à cet article précis, ou n'ajoute rien du "
        "tout plutôt que de combler avec une formule vide.\n"
        "- Varie la façon de démarrer l'accroche et la phrase de clôture à chaque génération : parfois "
        "une observation concrète, parfois une mise en situation, parfois une question directe posée à "
        "l'acheteur — ne reproduis jamais la même structure de phrase deux fois si on te redemande une "
        "version.\n"
        "- N'invente JAMAIS de détail factuel (marque, taille, défaut, matière) qui n'est ni fourni ni "
        "visible sur une éventuelle photo — reste vague plutôt que d'inventer.\n\n"
        "Consignes de conformité :\n"
        "- Ne mentionne jamais de contact ou de paiement en dehors de l'application (pas de téléphone, "
        "WhatsApp, Snapchat, PayPal, virement, lien externe).\n"
        "- Pas de fausses promotions ni de superlatifs trompeurs.\n\n"
        f"Ton à adopter : {ton_description}"
    )


def _parser_reponse_annonce(contenu: str, mots_cles: str):
    contenu = (contenu or "").strip()

    # Sécurité si le modèle entoure quand même sa réponse de ```json ... ```
    if contenu.startswith("```"):
        contenu = contenu.strip("`")
        if contenu.lower().startswith("json"):
            contenu = contenu[4:]
        contenu = contenu.strip()

    # 1) tentative JSON strict (cas normal)
    try:
        data = json.loads(contenu)
        titre = (data.get("titre") or "").strip()
        description = (data.get("description") or "").strip()
        if titre and description:
            return titre[:100], description
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # 2) filet de sécurité : le modèle a répondu en texte libre du style "Titre : ... / Description : ..."
    match_titre = re.search(r"titre\s*[:\-]\s*(.+)", contenu, re.IGNORECASE)
    match_description = re.search(r"description\s*[:\-]\s*([\s\S]+)", contenu, re.IGNORECASE)
    if match_titre and match_description:
        titre = match_titre.group(1).strip().split("\n")[0].strip(' "')
        description = match_description.group(1).strip().strip(' "')
        if titre and description:
            return titre[:100], description

    # 3) dernier recours : tout le texte comme description
    return mots_cles.title()[:100], contenu


async def _generer_annonce(mots_cles: str, details: Optional[str], ton_key: str, channel_id, photo: Optional[discord.Attachment]):
    """Appelle Groq et retourne (titre, description). Lève une exception en cas d'échec de l'appel."""
    texte_utilisateur = f"Mots-clés : {mots_cles.strip()}"
    if details and details.strip():
        texte_utilisateur += f"\nInfos complémentaires : {details.strip()}"

    async def _appel(avec_photo: bool) -> str:
        _verifier_groq_disponible()
        instruction_systeme = _construire_instruction_annonce(ton_key, channel_id, avec_photo)
        if avec_photo:
            contenu_msg = [
                {"type": "text", "text": texte_utilisateur},
                {"type": "image_url", "image_url": {"url": photo.url}},
            ]
            modele = MODELE_VISION
        else:
            contenu_msg = texte_utilisateur
            modele = MODELE_REDACTION

        appel_kwargs = dict(
            model=modele,
            max_tokens=1200,  # marge suffisante pour le raisonnement interne + la réponse sur les modèles raisonneurs
            temperature=0.9,  # un peu de créativité en plus pour éviter un texte trop robotique/répétitif
            messages=[
                {"role": "system", "content": instruction_systeme},
                {"role": "user", "content": contenu_msg},
            ],
        )
        options = _options_raisonnement(modele)
        try:
            # Le mode JSON strict force le modèle à respecter le format demandé (supporté par Groq).
            reponse = await asyncio.to_thread(
                groq_client.chat.completions.create,
                response_format={"type": "json_object"},
                **{**appel_kwargs, **options},
            )
        except Exception:
            # Si le modèle/l'API ne supporte pas un de ces paramètres, on retente sans.
            try:
                reponse = await asyncio.to_thread(groq_client.chat.completions.create, **{**appel_kwargs, **options})
            except Exception:
                reponse = await asyncio.to_thread(groq_client.chat.completions.create, **appel_kwargs)
        return (reponse.choices[0].message.content or "").strip()

    try:
        contenu = await _appel(avec_photo=photo is not None)
    except Exception:
        if photo is not None:
            # Le modèle vision peut être indisponible : on retente en texte seul avant d'abandonner.
            contenu = await _appel(avec_photo=False)
        else:
            raise

    if not contenu.strip():
        # Le modèle a tout consommé en raisonnement interne sans produire de réponse visible :
        # on retente une fois en texte seul, cette fois ça passe presque toujours.
        contenu = await _appel(avec_photo=False)

    return _parser_reponse_annonce(contenu, mots_cles)


async def _envoyer_annonce(
    interaction: discord.Interaction,
    mots_cles: str,
    details: Optional[str],
    ton_key: str,
    photo: Optional[discord.Attachment],
):
    try:
        titre, description = await _generer_annonce(mots_cles, details, ton_key, interaction.channel_id, photo)
    except Exception as e:
        await interaction.followup.send(f"❌ {_erreur_ia_lisible(e)}", ephemeral=True)
        return

    embed = discord.Embed(title="📋 Annonce générée", color=discord.Color.from_rgb(255, 105, 180))
    embed.timestamp = discord.utils.utcnow()
    embed.add_field(name="✏️ Titre", value=f"**{titre}**", inline=False)
    embed.add_field(name="📄 Description", value=f"```{description}```", inline=False)
    if photo:
        embed.set_thumbnail(url=photo.url)
    embed.set_footer(
        text=f"Ton : {TONS_ANNONCE_LABELS.get(ton_key, ton_key)} • Généré par IA, à relire avant publication ✅"
    )

    vue = AnnonceResultView(mots_cles, details, ton_key, photo)
    vue.message = await interaction.followup.send(embed=embed, view=vue, ephemeral=True)


class AnnonceResultView(discord.ui.View):
    """Vue avec un bouton pour régénérer une nouvelle proposition d'annonce."""

    def __init__(self, mots_cles: str, details: Optional[str], ton_key: str, photo: Optional[discord.Attachment]):
        super().__init__(timeout=600)
        self.mots_cles = mots_cles
        self.details = details
        self.ton_key = ton_key
        self.photo = photo
        self.message = None  # assigné après l'envoi, pour pouvoir désactiver le bouton à l'expiration

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass

    @discord.ui.button(label="Régénérer", style=discord.ButtonStyle.secondary, emoji="🔄")
    async def regenerer(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True, ephemeral=True)
        await _envoyer_annonce(interaction, self.mots_cles, self.details, self.ton_key, self.photo)


class AnnonceModal(discord.ui.Modal, title="📝 Générateur de description Vinted"):
    mots_cles = discord.ui.TextInput(
        label="Mots-clés (3-4 minimum)",
        placeholder="ex: pull, nike, bleu, taille M",
        required=True,
        max_length=200,
    )
    details = discord.ui.TextInput(
        label="Détails complémentaires (optionnel)",
        style=discord.TextStyle.paragraph,
        placeholder="marque, matière, mesures, état, défauts éventuels...",
        required=False,
        max_length=500,
    )

    def __init__(self, photo: Optional[discord.Attachment] = None, ton_key: str = "accrocheur"):
        super().__init__()
        self.photo = photo
        self.ton_key = ton_key

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        await _envoyer_annonce(interaction, self.mots_cles.value, self.details.value, self.ton_key, self.photo)


class AnnonceIntroView(discord.ui.View):
    def __init__(self, photo: Optional[discord.Attachment] = None, ton_key: str = "accrocheur"):
        super().__init__(timeout=None)
        self.photo = photo
        self.ton_key = ton_key

    @discord.ui.button(label="Générer la description", style=discord.ButtonStyle.success, emoji="📝", custom_id="annonce_lancer_bouton")
    async def lancer(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(AnnonceModal(photo=self.photo, ton_key=self.ton_key))


@bot.tree.command(name="description", description="Génère un titre + une description Vinted à partir de mots-clés")
@app_commands.describe(
    photo="Photo de l'article (optionnel, aide l'IA à préciser couleur/matière/état)",
    ton="Ton de la description (par défaut : accrocheur)",
)
@app_commands.choices(ton=[
    app_commands.Choice(name="✨ Accrocheur (par défaut)", value="accrocheur"),
    app_commands.Choice(name="📝 Sobre et factuel", value="sobre"),
    app_commands.Choice(name="😄 Fun / familier", value="fun"),
    app_commands.Choice(name="🎭 Utiliser ma personnalité actuelle (/style)", value="personnalite"),
])
async def description(
    interaction: discord.Interaction,
    photo: Optional[discord.Attachment] = None,
    ton: Optional[app_commands.Choice[str]] = None,
):
    ton_key = ton.value if ton else "accrocheur"
    embed = discord.Embed(
        title="📝 Générateur de description Vinted",
        description=(
            "Donne quelques mots-clés, et l'IA te rédige un titre + une description complète, "
            "prête à coller sur Vinted.\n\n"
            "**Comment ça marche ?**\n"
            "1️⃣ Clique sur le bouton ci-dessous\n"
            "2️⃣ Donne 3-4 mots-clés (ex: pull, nike, bleu, taille M)\n"
            "3️⃣ Ajoute des détails si tu veux (marque, matière, défauts...)\n"
            "4️⃣ Si tu as joint une photo à la commande, l'IA s'en sert pour affiner la description."
        ),
        color=discord.Color.from_rgb(255, 105, 180),
    )
    if photo:
        embed.set_thumbnail(url=photo.url)
    embed.set_footer(text=f"Ton sélectionné : {TONS_ANNONCE_LABELS.get(ton_key, ton_key)} • Généré par IA, à relire avant publication.")
    view = AnnonceIntroView(photo=photo, ton_key=ton_key)
    await interaction.response.send_message(embed=embed, view=view)


# ============================================================
#  Commande /recherche — parcourir des annonces Vinted (achat)
# ============================================================

def _signaux_vigilance(prix, favoris, prix_median):
    """Détecte quelques signaux simples (pas une certitude, juste des indices à vérifier soi-même)
    à partir des données déjà publiques de l'annonce : prix très en dessous du marché, ou annonce
    encore sans aucun favori."""
    signaux = []
    if prix is not None and prix_median and prix < prix_median * 0.4:
        signaux.append("💸 Prix nettement en dessous du marché — vérifie bien les photos et le profil du vendeur avant de payer.")
    if not favoris:
        signaux.append("👀 Aucun favori pour l'instant — une annonce très récente ou peu visible, reste prudent.")
    return signaux


class RechercheGalerieView(GalerieView):
    """Galerie de résultats de /recherche : mêmes flèches que /estimer, plus un signal de vigilance
    par annonce et un bouton pour transformer directement la recherche en alerte de prix."""

    def __init__(self, items, embed_principal, requete: str, prix_max: Optional[float], prix_median: Optional[float]):
        self.prix_median = prix_median
        super().__init__(items, embed_principal)
        self.requete = requete
        self.prix_max = prix_max

    def _embed_item_courant(self) -> discord.Embed:
        e = super()._embed_item_courant()
        i = self.items[self.index]
        signaux = _signaux_vigilance(_prix_de(i), _favoris_de(i), self.prix_median)
        if signaux:
            e.add_field(name="⚠️ À vérifier avant d'acheter", value="\n".join(signaux), inline=False)
        return e

    @discord.ui.button(label="Créer une alerte pour cette recherche", style=discord.ButtonStyle.primary, emoji="🔔", row=1)
    async def creer_alerte(self, interaction: discord.Interaction, button: discord.ui.Button):
        ok, message = await _creer_alerte(interaction.user.id, self.requete, self.prix_max)
        await interaction.response.send_message(message, ephemeral=True)


async def _lancer_recherche(interaction: discord.Interaction, requete_texte: str, prix_max_texte: Optional[str]):
    requete = requete_texte.strip()
    prix_max = None
    if prix_max_texte and prix_max_texte.strip():
        try:
            prix_max = float(prix_max_texte.strip().replace(",", "."))
        except ValueError:
            await interaction.followup.send(
                "⚠️ Le prix maximum doit être un nombre (ex: 30 ou 29.90). Recherche relancée sans filtre de prix.",
                ephemeral=True,
            )
            prix_max = None

    try:
        items = await _rechercher_vinted_cache(requete, per_page=50)
    except asyncio.TimeoutError:
        await interaction.followup.send("⏱️ Vinted met trop de temps à répondre. Réessaie dans quelques minutes.", ephemeral=True)
        return
    except VintedAuthError:
        await interaction.followup.send(
            "🚫 Vinted a temporairement bloqué la connexion (ça arrive régulièrement avec les hébergeurs gratuits). "
            "Ce n'est pas systématique — réessaie dans quelques minutes, ça passe souvent au 2e ou 3e essai.",
            ephemeral=True,
        )
        return
    except VintedRateLimitError:
        await interaction.followup.send(
            "🚫 Vinted a temporairement limité les requêtes (trop de recherches d'un coup). Réessaie dans quelques minutes.",
            ephemeral=True,
        )
        return
    except (VintedNetworkError, VintedAPIError, VintedError) as e:
        await interaction.followup.send(f"❌ Erreur Vinted : `{e}`", ephemeral=True)
        return
    except Exception as e:
        await interaction.followup.send(f"❌ Erreur inattendue pendant la recherche : `{e}`", ephemeral=True)
        return

    if prix_max is not None:
        items = [i for i in items if (_prix_de(i) or 0) <= prix_max]

    embed_principal = discord.Embed(
        title="🛍️ Résultats de recherche Vinted",
        color=discord.Color.from_rgb(255, 140, 0),
    )
    embed_principal.timestamp = discord.utils.utcnow()
    embed_principal.add_field(name="🔎 Recherche", value=f"**{requete}**", inline=False)
    if prix_max is not None:
        embed_principal.add_field(name="💶 Prix max", value=f"{prix_max:.2f} €", inline=True)
    embed_principal.add_field(name="📦 Annonces trouvées", value=str(len(items)), inline=True)

    if not items:
        embed_principal.description = "Aucune annonce ne correspond à cette recherche pour le moment. Essaie une description plus générale."
        await interaction.followup.send(embed=embed_principal, ephemeral=True)
        return

    prix_connus = [p for p in (_prix_de(i) for i in items) if p is not None]
    prix_median = statistics.median(prix_connus) if prix_connus else None

    top = items[:15]
    vue = RechercheGalerieView(top, embed_principal, requete, prix_max, prix_median)
    vue.message = await interaction.followup.send(
        embeds=[embed_principal, vue._embed_item_courant()], view=vue, ephemeral=True
    )


class RechercheModal(discord.ui.Modal, title="🛍️ Nouvelle recherche Vinted"):
    requete = discord.ui.TextInput(
        label="Que cherches-tu ?",
        placeholder="ex: nike air force taille 42",
        required=True,
        max_length=150,
    )
    prix_max = discord.ui.TextInput(
        label="Prix maximum en € (optionnel)",
        placeholder="ex: 30",
        required=False,
        max_length=10,
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        await _lancer_recherche(interaction, self.requete.value, self.prix_max.value)


class RechercheIntroView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Rechercher", style=discord.ButtonStyle.success, emoji="🛍️", custom_id="recherche_lancer_bouton")
    async def lancer(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(RechercheModal())


@bot.tree.command(name="recherche", description="Parcourt des annonces Vinted correspondant à ta recherche")
async def recherche(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🛍️ Recherche Vinted",
        description=(
            "Donne ce que tu cherches, et le bot te ramène les annonces Vinted correspondantes, "
            "à parcourir directement ici avec les flèches.\n\n"
            "**Comment ça marche ?**\n"
            "1️⃣ Clique sur le bouton ci-dessous\n"
            "2️⃣ Décris ce que tu cherches (ex: nike air force taille 42)\n"
            "3️⃣ Indique un prix maximum si tu veux filtrer (optionnel)\n"
            "4️⃣ Parcours les annonces trouvées avec les flèches ◀ ▶, chacune avec un lien direct, "
            "un signal ⚠️ si le prix semble anormalement bas ou l'annonce très peu vue\n"
            "5️⃣ Rien ne te convient encore ? Clique sur 🔔 pour transformer ta recherche en alerte "
            "et être prévenu(e) dès qu'une bonne annonce correspondante apparaît."
        ),
        color=discord.Color.from_rgb(255, 140, 0),
    )
    embed.set_footer(text="Données publiques Vinted, à titre indicatif.")
    view = RechercheIntroView()
    await interaction.response.send_message(embed=embed, view=view)


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Variable d'environnement DISCORD_TOKEN manquante.")
    bot.run(token)
