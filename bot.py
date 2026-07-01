import os
import statistics
import asyncio
from urllib.parse import quote
import discord
from discord import app_commands
from discord.ext import commands
from vinted import VintedClient, VintedRateLimitError, VintedNetworkError, VintedAPIError, VintedError

# --- Configuration de base ---
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


# --- Le bouton "Accès au lien" ---
class LinkButtonView(discord.ui.View):
    def __init__(self, lien: str):
        super().__init__(timeout=None)  # le bouton reste actif indéfiniment
        self.lien = lien

    @discord.ui.button(label="Accès au lien", style=discord.ButtonStyle.primary, emoji="🔗")
    async def reveal_link(self, interaction: discord.Interaction, button: discord.ui.Button):
        # ephemeral=True => seul celui qui clique voit le message
        await interaction.response.send_message(
            f"🔗 **Lien :** {self.lien}",
            ephemeral=True
        )


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


GUILD_ID = os.environ.get("GUILD_ID")


@bot.event
async def on_ready():
    try:
        if GUILD_ID:
            # Synchronisation sur un seul serveur = instantanée (pas d'attente d'1h)
            guild = discord.Object(id=int(GUILD_ID))
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
        else:
            synced = await bot.tree.sync()
        print(f"{len(synced)} commande(s) synchronisée(s).")
    except Exception as e:
        print(f"Erreur de synchronisation : {e}")
    print(f"Connecté en tant que {bot.user} — bot prêt.")


# --- La commande slash /poster ---
@bot.tree.command(name="poster", description="Crée une annonce avec titre, texte encadré et bouton lien caché")
@app_commands.describe(
    titre="Le titre affiché en haut de l'annonce",
    texte="Le texte principal (affiché dans le cadre)",
    lien="Le lien qui sera révélé uniquement au clic (visible seulement par la personne qui clique)"
)
async def poster(interaction: discord.Interaction, titre: str, texte: str, lien: str):
    embed = discord.Embed(
        title=titre,
        description=texte,
        color=discord.Color.blurple()
    )
    view = LinkButtonView(lien)
    await interaction.response.send_message(embed=embed, view=view)


# --- La commande slash /estimer ---
@bot.tree.command(name="estimer", description="Estime le prix de vente d'un article à partir d'annonces Vinted comparables")
@app_commands.describe(
    article="Description courte de l'article (ex: pull zara laine col rond)",
    marque="Marque de l'article (optionnel, améliore la précision)",
    taille="Taille de l'article (optionnel)",
    photo="Photo de l'article à vendre (optionnel, juste pour l'affichage)"
)
async def estimer(
    interaction: discord.Interaction,
    article: str,
    marque: str = None,
    taille: str = None,
    photo: discord.Attachment = None,
):
    # La recherche peut prendre quelques secondes, on prévient Discord qu'on répondra plus tard
    await interaction.response.defer(thinking=True)

    requete = " ".join(filter(None, [marque, article, taille]))
    print(f"[estimer] requête envoyée à Vinted : {requete!r}")

    try:
        url = f"https://www.vinted.fr/catalog?search_text={quote(requete)}&order=relevance"

        async def _rechercher():
            async with VintedClient() as client:
                return await client.search_items(url=url, per_page=50, raw_data=True)

        # Maximum 20 secondes d'attente
        items = await asyncio.wait_for(_rechercher(), timeout=20)
        print(f"[estimer] nb_items={len(items) if items else 0}")

    except asyncio.TimeoutError:
        await interaction.followup.send(
            "⏱️ Vinted met trop de temps à répondre. Réessaie dans quelques minutes."
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
        await interaction.followup.send(
            f"❌ Erreur inattendue pendant la recherche : `{e}`"
        )
        return

    if not items:
        await interaction.followup.send("Aucune annonce comparable trouvée. Essaie une description plus générale.")
        return

    # Nettoyage des prix (parfois vides ou mal formatés)
    prix_valides = []
    for i in items:
        p = _prix_de(i)
        if p is not None:
            prix_valides.append(p)

    if not prix_valides:
        await interaction.followup.send("Impossible de récupérer des prix exploitables sur ces annonces.")
        return

    favoris = [_champ(i, "favourite_count", 0) or 0 for i in items]

    prix_moyen = statistics.mean(prix_valides)
    prix_median = statistics.median(prix_valides)
    prix_min = min(prix_valides)
    prix_max = max(prix_valides)
    demande_moyenne = statistics.mean(favoris) if favoris else 0

    # Prix conseillé : légèrement sous le médian pour vendre plus vite,
    # jamais en dessous du minimum observé
    prix_conseille = max(round(prix_median * 0.95, 2), prix_min)

    # Les 3 annonces comparables qui ont le plus de favoris = les plus "demandées"
    top = sorted(items, key=lambda i: _champ(i, "favourite_count", 0) or 0, reverse=True)[:3]

    embed = discord.Embed(
        title=f"📊 Estimation — {article}",
        description=f"Basé sur {len(items)} annonces comparables trouvées sur Vinted",
        color=discord.Color.green(),
    )
    if photo:
        embed.set_thumbnail(url=photo.url)
    embed.add_field(name="Prix moyen", value=f"{prix_moyen:.2f} €", inline=True)
    embed.add_field(name="Prix médian", value=f"{prix_median:.2f} €", inline=True)
    embed.add_field(name="Fourchette", value=f"{prix_min:.2f} € – {prix_max:.2f} €", inline=True)
    embed.add_field(name="❤️ Favoris moyens (demande)", value=f"{demande_moyenne:.1f}", inline=True)
    embed.add_field(name="💡 Prix conseillé pour vendre vite", value=f"**{prix_conseille:.2f} €**", inline=False)

    if top:
        lignes = []
        for i in top:
            titre = (_champ(i, "title") or "Sans titre")[:45]
            url = _champ(i, "url", "")
            prix_affiche = _prix_de(i)
            favs = _champ(i, "favourite_count", 0) or 0
            lignes.append(f"[{titre}]({url}) — {prix_affiche} € — ❤️ {favs}")
        embed.add_field(name="Annonces les plus populaires (référence)", value="\n".join(lignes), inline=False)

    embed.set_footer(text="Données publiques Vinted, à titre indicatif.")

    await interaction.followup.send(embed=embed)


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Variable d'environnement DISCORD_TOKEN manquante.")
    bot.run(token)
