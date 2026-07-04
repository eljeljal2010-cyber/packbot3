import os
import statistics
import asyncio
from typing import Optional
from pathlib import Path
from urllib.parse import quote
import discord
from discord import app_commands
from discord.ext import commands
from vinted import (
    VintedClient,
    VintedRateLimitError,
    VintedNetworkError,
    VintedAPIError,
    VintedAuthError,
    VintedError,
)

# --- Configuration de base ---
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

GUILD_ID = os.environ.get("GUILD_ID")


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
    def __init__(self, lien: str):
        super().__init__(timeout=None)
        self.lien = lien

    @discord.ui.button(label="Accès au lien", style=discord.ButtonStyle.primary, emoji="🔗")
    async def reveal_link(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message(
            f"🔗 **Lien :** {self.lien}",
            ephemeral=True
        )


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
    print(f"Connecté en tant que {bot.user} — bot prêt.")


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
    embed = discord.Embed(title=titre, description=texte, color=discord.Color.blurple())
    view = LinkButtonView(lien)
    await interaction.response.send_message(embed=embed, view=view)


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
        self._maj_etat_boutons()

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
        url = f"https://www.vinted.fr/catalog?search_text={quote(requete)}&order=relevance"

        async def _rechercher():
            derniere_erreur = None
            for tentative in range(3):
                try:
                    async with VintedClient(
                        persist_cookies=True,
                        cookies_dir=Path("/tmp/vinted_cookies"),
                    ) as client:
                        return await client.search_items(url=url, per_page=50, raw_data=True)
                except (VintedAuthError, VintedRateLimitError) as e:
                    derniere_erreur = e
                    if tentative < 2:
                        await asyncio.sleep(4 * (tentative + 1))
            raise derniere_erreur

        items = await asyncio.wait_for(_rechercher(), timeout=45)
        print(f"[estimer] nb_items={len(items) if items else 0}")

    except asyncio.TimeoutError:
        await interaction.followup.send("⏱️ Vinted met trop de temps à répondre. Réessaie dans quelques minutes.")
        return
    except VintedAuthError:
        await interaction.followup.send(
            "🚫 Vinted a temporairement bloqué la connexion (ça arrive régulièrement avec les hébergeurs gratuits). "
            "Ce n'est pas systématique — réessaie dans quelques minutes, ça passe souvent au 2e ou 3e essai."
        )
        return
    except VintedRateLimitError:
        await interaction.followup.send(
            "🚫 Vinted a temporairement limité les requêtes (trop de recherches d'un coup). "
            "Réessaie dans quelques minutes."
        )
        return
    except (VintedNetworkError, VintedAPIError, VintedError) as e:
        await interaction.followup.send(f"❌ Erreur Vinted : `{e}`")
        return
    except Exception as e:
        await interaction.followup.send(f"❌ Erreur inattendue pendant la recherche : `{e}`")
        return

    if not items:
        await interaction.followup.send("Aucune annonce comparable trouvée. Essaie une description plus générale.")
        return

    # --- Prix ---
    prix_bruts = [p for p in (_prix_de(i) for i in items) if p is not None]
    if not prix_bruts:
        await interaction.followup.send("Impossible de récupérer des prix exploitables sur ces annonces.")
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
        await interaction.followup.send(embeds=[embed_principal, vue._embed_item_courant()], view=vue)
    else:
        await interaction.followup.send(embed=embed_principal)


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
        await interaction.response.defer(thinking=True)

        prix_achat_valeur = None
        if self.prix_achat.value:
            try:
                prix_achat_valeur = float(self.prix_achat.value.replace(",", "."))
            except ValueError:
                await interaction.followup.send(
                    f"⚠️ Le prix d'achat `{self.prix_achat.value}` n'est pas un nombre valide, "
                    "il a été ignoré pour cette estimation."
                )

        marge_cible_valeur = None
        if self.marge_cible.value:
            try:
                marge_cible_valeur = float(self.marge_cible.value.replace(",", "."))
            except ValueError:
                await interaction.followup.send(
                    f"⚠️ La marge `{self.marge_cible.value}` n'est pas un nombre valide, "
                    "elle a été ignorée (calcul automatique utilisé)."
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
        super().__init__(timeout=300)
        self.photo = photo

    @discord.ui.button(label="Lancer une estimation", style=discord.ButtonStyle.success, emoji="🔍")
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


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Variable d'environnement DISCORD_TOKEN manquante.")
    bot.run(token)
