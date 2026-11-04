"""
Automated deployment script for Hugging Face Spaces (Docker).
Allows pushing the project to Hugging Face Cloud in one single command.
"""

import sys
import subprocess
import shutil

def run_cmd(cmd):
    print(f"[*] Running: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"[!] Error: {res.stderr.strip()}")
        return False, res.stderr
    print(f"[+] Output: {res.stdout.strip()}")
    return True, res.stdout

def main():
    print("=" * 65)
    print("      DEPLOIEMENT CLOUD - HUGGING FACE SPACES (DOCKER)")
    print("=" * 65)
    print("\nCe script va pousser votre projet sur votre Hugging Face Space.")
    print("1. Creez un Space gratuit sur: https://huggingface.co/new-space")
    print("   - SDK: Choisissez 'Docker' (Blank)")
    print("   - Visibilite: Public\n")

    if len(sys.argv) > 1:
        space_url = sys.argv[1].strip()
    else:
        print("Usage: python deploy_hf.py <URL_DE_VOTRE_SPACE_OU_REPO>")
        print("Exemple: python deploy_hf.py https://huggingface.co/spaces/monuser/youtube-sentiment")
        space_url = input("\nCollez l'URL Git de votre Hugging Face Space : ").strip()

    if not space_url:
        print("[!] Aucune URL fournie. Annulation.")
        return

    # Clean URL format
    if not space_url.endswith(".git") and "huggingface.co" in space_url:
        target_remote = space_url
    else:
        target_remote = space_url

    print(f"\n[*] Configuration du remote Git vers: {target_remote}")
    run_cmd("git remote remove space")
    success, _ = run_cmd(f"git remote add space {target_remote}")
    if not success:
        print("[!] Impossible d'ajouter le remote Git.")
        return

    print("\n[*] Poussee du code et du modele vers Hugging Face Spaces...")
    success, out = run_cmd("git push space master:main --force")
    if success:
        print("\n" + "=" * 65)
        print("[+] SUCCES ! Votre projet est en cours de deploiement sur le Cloud !")
        print("Consultez l'onglet 'App' de votre Space pour voir le conteneur demarrer.")
        print("Une fois en ligne, copiez l'URL HTTPS directe et collez-la dans")
        print("les parametres (engrenage) de votre extension Chrome !")
        print("=" * 65)
    else:
        print("\n[!] Astuce d'authentification Git :")
        print("Si Git demande un mot de passe, utilisez votre User Access Token Hugging Face")
        print("(Settings -> Access Tokens -> New Token -> Role: Write) :")
        print("https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    main()
