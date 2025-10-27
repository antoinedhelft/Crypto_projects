-- Crée le rôle s'il n'existe pas
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='crypto') THEN
    CREATE ROLE crypto LOGIN PASSWORD 'crypto';
  END IF;
END$$;

-- Création des bases (cluster neuf -> succès)
CREATE DATABASE airflow_meta;
CREATE DATABASE crypto_trading OWNER crypto;

-- Donner les droits / ownership schéma public sur crypto_trading
\connect crypto_trading
ALTER SCHEMA public OWNER TO crypto;
GRANT ALL ON SCHEMA public TO crypto;
GRANT ALL ON DATABASE crypto_trading TO crypto;