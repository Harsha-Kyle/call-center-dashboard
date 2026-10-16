# SecureVault — File Encryption & Decryption Dashboard

A Dash web app for compressing files, encrypting them with AES-256-GCM, and restoring them using an encryption key.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8050`.

## Deploy to Render

1. Push `app.py`, `requirements.txt`, and `render.yaml` to a GitHub repository.
2. In Render, choose **New → Blueprint** and connect the repository, or create a **Web Service** manually.
3. If using a manual Web Service, use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:server --workers 1 --threads 4 --timeout 120`
4. Render will build and provide a public URL.

The included `render.yaml` configures a free web service. Free services may sleep when idle and take time to wake up. Availability, quotas, and plan terms can change.

## File limits

- Upload limit defaults to 8 MB per file (`MAX_FILE_SIZE_MB`).
- Decompressed output limit defaults to 64 MB (`MAX_DECOMPRESSED_SIZE_MB`).
- You can change these environment variables in Render, but increasing limits also increases memory and resource use.

## Security notes

- AES-256-GCM authenticates encrypted content before it is accepted.
- The app creates a new random key for every encryption. Download and store the key separately; losing it means the file cannot be recovered.
- The downloadable key is saved as 64 hexadecimal characters in a text file. The app also accepts legacy raw 32-byte key files.
- Uploaded files are sent to the server for processing. Do not use a public demo for confidential or highly sensitive documents. This project does not provide persistent storage or user accounts.
- This is a portfolio/demo deployment, not a substitute for a security audit. Consider client-side encryption if files must never leave the user's device.

## GitHub hygiene

Do not commit private files, real encryption keys, or confidential documents. Keep `.env`, local virtual environments, and Python cache files out of Git.
