# Rapport AI-SmartFactory

## Contenu

- `main.tex` : source LaTeX du rapport.
- `main.pdf` : version compilée.
- `figures/` : copie des figures disponibles utilisées ou citées dans le rapport.
- `PROJECT_AUDIT.md` : audit factuel du dépôt et des fonctionnalités.

Les figures sont regroupées dans `report/figures/` afin que le rapport soit
autonome. Les originaux restent dans le dossier `figures/` du projet.

## Compilation

Depuis le dossier `report/` :

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

La compilation nécessite une distribution LaTeX fonctionnelle, par exemple
MiKTeX, ainsi que les figures présentes dans `report/figures/`. Depuis la racine
du projet :

```powershell
Set-Location report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Personnalisation avant dépôt

Compléter le nom de l'encadrant sur la page de garde et adapter l'année
universitaire si nécessaire. Aucune capture d'écran du dashboard n'étant fournie
dans le dépôt, le chapitre 8 décrit le code réellement observé sans en inventer.
Les notebooks, les données et l'application ne sont pas modifiés par la
génération du rapport.
