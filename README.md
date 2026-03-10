# SSSIHL Knowledge Assistant

This is a Retrieval-Augmented Generation (RAG) assistant for Sri Sathya Sai Institute of Higher Learning, built with Streamlit, Pinecone, and OpenRouter for access to multiple free, powerful AI models.

## How to deploy on Streamlit Community Cloud

1. **Push your code to GitHub**
   * Create a new empty repository on GitHub.
   * On your computer, run standard Git commands to push your local code to this repo:
     ```bash
     git add .
     git commit -m "Ready for Streamlit Cloud"
     git branch -M main
     git remote add origin https://github.com/yourusername/your-repo-name.git
     git push -u origin main
     ```

2. **Deploy on Streamlit Cloud**
   * Go to [share.streamlit.io](https://share.streamlit.io/).
   * Click **Create app** and follow the prompts to sign in with GitHub.
   * **Repository**: Select your newly created repo.
   * **Main file path**: Type `app.py`.
   * **App URL**: Pick a custom URL (optional).

3. **Configure Secrets**
   * Before clicking "Deploy", click on **Advanced settings**.
   * In the **Secrets** section, you can optionally paste your API keys so users don't have to enter them every time:
     ```toml
     OPENROUTER_API_KEY = "sk-or-v1-..."
     PINECONE_API_KEY = "pcsk_..."
     PINECONE_INDEX = "saiinst"
     ```
   * Click **Save** and then **Deploy!**

Your app should be live within minutes! 🚀
