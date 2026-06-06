
## Variables d'environnement et démarrage

Version fonctionnelle :

```
1) Clonez le repo
```
Utilisation de python 3.12.9 via pyenv-win
```
2) Puis créez un fichier `.env` en vous basant sur le `.env.example` en modifiant avec vos valeurs locales si besoin. (C'est juste un copier/coller en modifiant avec les valeurs que vous souhaitez)

- Pour obtenir la clé airflow secrète à mettre dans .env :
python -c "import secrets; print(secrets.token_urlsafe(64))"

- Pour obtenir la clé Fernet à mettre dans .env :
docker compose run --rm airflow-webserver python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

(ps : si vous avez déjà build les conteneurs, vous pouvez juste relancer ceux utiles :
docker compose up -d --build airflow-webserver airflow-scheduler)

3) Créez un venv pour les dépendances/bibliothèques
```
- python -m venv .venv
- .\.venv\Scripts\Activate
- pip install -r requirements_local.txt
``
4) Lancez docker desktop

5) Puis dans votre IDE lancez ces commandes :
`docker compose --profile images build` # build des images
`docker compose up -d --build` # build des conteneurs


Permet de démarrer l'ensemble automatiquement, plus besoin d'intéragir avec airflow, même lors du premier lancement.


###
Si vous modifiez le data pipeline ou ml pipeline, il faut recosntruire l'image :
`docker build -t crypto_data_pipeline:latest -f docker/Dockerfile.data_pipeline .`
`docker build -t crypto_ml_pipeline:latest -f docker/Dockerfile.ml_pipeline .`
Et redémarrer les service Airflow.

- Une fois lancez, vous pourrez accéder à fastapi à l'adresse : 
localhost:8000/docs
- airflow : 
localhost:8080    ### les identifiants dans .env sont admin/admin de base, c'est pourquoi vous pouvez/devez les changer
- streamlit : 
localhost:8501


- Pour éteindre docker sans perte de données :
`docker compose down`

- Pour relancer docker : 
`docker compose up -d`

# Rebuild ciblé après modifications importantes:
- Si vous modifiez l'API ou ses dépendances: `docker compose build api ; docker compose up -d api`
- Si vous modifiez Airflow (DAGs ou image): `docker compose build airflow ; docker compose up -d airflow`
- Pour (re)charger uniquement les DAGs sans rebuild d'image, ils sont montés dans le conteneur: un simple `docker compose up -d airflow-webserver airflow-scheduler` suffit après vos changements de fichiers dans `airflow/dags/`.

# Accéder à la base Postgres dans le conteneur:
`docker exec -it pg_crypto psql -U crypto -d crypto_trading`

ou utiliser l'extension vscode 'PostgreSQL' de Chris Kolkman (F5 pour run la query)


### Si vous aviez déjà lancé le projet :
`docker compose down --volumes --remove-orphans`
Permet d'arrêter et nettoyer les conteneur, réseaux et volumes du projets
# Supprimez toutes les images directement via docker desktop, sinon :
`docker compose down --rmi local --volumes --remove-orphans`
`docker system prune -a --volumes`
# ATTENTION ça purge toutes les images, même celles qui ne sont pas du projets si elles sont taguées pareil.

Puis on rebuild complètement :
`docker compose build --no-cache`

puis on démarre :
`docker compose up -d`