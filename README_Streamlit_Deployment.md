# AI Candidate Screening Agent

A Streamlit UI for the LangChain + LangGraph candidate-screening workflow.

## Deployment

1. Push `app.py` and `requirements.txt` to GitHub.
2. Open https://share.streamlit.io/
3. Sign in with GitHub.
4. Choose Create app.
5. Select the repository and branch.
6. Set the main file to `app.py`.
7. Deploy.
8. In the app settings, add this secret:

```toml
OPENAI_API_KEY = "your-new-openai-api-key"
```

Never commit the actual API key to GitHub.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```
