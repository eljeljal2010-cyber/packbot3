import os
import discord
from discord import app_commands
from discord.ext import commands

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


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Variable d'environnement DISCORD_TOKEN manquante.")
    bot.run(token)
