import os
import statistics
import discord
from discord import app_commands
from discord.ext import commands
from vinted import Vinted

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


@bot.event
async def on_ready():
    try:
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
    photo="Photo de l'article à vendre (affichée avec le résultat)",
    article="Description courte de l'article (ex: pull zara laine col rond)",
    marque="Marque de l'article (optionnel, améliore la précision)",
    taille="Taille de l'article (optionnel)"
)
async def estimer(
    interaction: discord.Interaction,
    photo: discord.Attachment,
    article: str,
    marque: str = None,
    taille: str = None,
):
    # La recherche peut prendre quelques secondes, on prévient Discord qu'on répondra plus tard
    await interaction.response.defer(thinking=True)

    requete = " ".join(filter(None, [marque, article, taille]))

    try:
        vinted = Vinted(domain="fr")
        resultat = vinted.search(query=requete, per_page=50)
        items = resultat.items
    except Exception as e:
        await interaction.followup.send(
            f"❌ Erreur pendant la recherche sur Vinted (le site a peut-être temporairement bloqué "
            f"la requête, réessaie dans quelques minutes) : `{e}`"
        )
        return

    if not items:
        await interaction.followup.send("Aucune annonce comparable trouvée. Essaie une description plus générale.")
        return

    # Nettoyage des prix (parfois vides ou mal formatés)
    prix_valides = []
    for i in items:
        try:
            prix_valides.append(float(i.price))
        except (TypeError, ValueError):
            continue

    if not prix_valides:
        await interaction.followup.send("Impossible de récupérer des prix exploitables sur ces annonces.")
        return

    favoris = [getattr(i, "favourite_count", 0) or 0 for i in items]

    prix_moyen = statistics.mean(prix_valides)
    prix_median = statistics.median(prix_valides)
    prix_min = min(prix_valides)
    prix_max = max(prix_valides)
    demande_moyenne = statistics.mean(favoris) if favoris else 0

    # Prix conseillé : légèrement sous le médian pour vendre plus vite,
    # jamais en dessous du minimum observé
    prix_conseille = max(round(prix_median * 0.95, 2), prix_min)

    # Les 3 annonces comparables qui ont le plus de favoris = les plus "demandées"
    top = sorted(items, key=lambda i: getattr(i, "favourite_count", 0) or 0, reverse=True)[:3]

    embed = discord.Embed(
        title=f"📊 Estimation — {article}",
        description=f"Basé sur {len(items)} annonces comparables trouvées sur Vinted",
        color=discord.Color.green(),
    )
    embed.set_image(url=photo.url)
    embed.add_field(name="Prix moyen", value=f"{prix_moyen:.2f} €", inline=True)
    embed.add_field(name="Prix médian", value=f"{prix_median:.2f} €", inline=True)
    embed.add_field(name="Fourchette", value=f"{prix_min:.2f} € – {prix_max:.2f} €", inline=True)
    embed.add_field(name="❤️ Favoris moyens (demande)", value=f"{demande_moyenne:.1f}", inline=True)
    embed.add_field(name="💡 Prix conseillé pour vendre vite", value=f"**{prix_conseille:.2f} €**", inline=False)

    if top:
        lignes = []
        for i in top:
            titre = (i.title or "Sans titre")[:45]
            lignes.append(f"[{titre}]({i.url}) — {i.price} € — ❤️ {getattr(i, 'favourite_count', 0)}")
        embed.add_field(name="Annonces les plus populaires (référence)", value="\n".join(lignes), inline=False)

    embed.set_footer(text="Données publiques Vinted, à titre indicatif.")

    await interaction.followup.send(embed=embed)


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Variable d'environnement DISCORD_TOKEN manquante.")
    bot.run(token)
