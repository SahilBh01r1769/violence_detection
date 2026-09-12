# Hosted demo branch

This disposable branch contains a Streamlit-only dashboard showcase with two
real application-output clips. It is separate from the FastAPI-backed runtime
in `main` so it can be hosted for a visual walkthrough without requiring a
webcam, RTSP source, or local API process.

To run it locally:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

For Streamlit Community Cloud, select this branch and set the app file to
`demo/app.py`. Delete the branch after the recording is complete.
