# Bluesky User Analysis Streamlit App

Example streamlit app that performs some analysis on a given Bluesky user. Built while exploring ATProto (what Bluesky runs on) and decided to publish it in case someone else may find it useful.

## Using

The app is built using [Streamlit](https://streamlit.io/). You can install dependencies using the `requirements.txt` file:

```
pip install -r requirements.txt
```

The app requires login credentials for a Bluesky app to perform the API queries. Create a .streamlit/secrets.toml file and add the following:

```toml
bluesky_login="YOUR_LOGIN"
bluesky_password="YOUR_PASS"
```

Then, run streamlit:

```
streamlit run bluesky_user_analysis.py
```

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/fpcorso/bluesky-streamlit-user-analysis/blob/main/LICENSE) for more details.
