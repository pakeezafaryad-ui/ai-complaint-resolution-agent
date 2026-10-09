# AI Complaint Resolution Agent

An AI-powered complaint management application built using:

- Python
- CrewAI
- Groq API
- Streamlit
- Pandas
- CSV and JSON storage

## Features

- Complaint submission form
- AI-assisted complaint analysis
- Multi-agent resolution recommendations
- Complaint tracking dashboard
- CSV report downloads
- Human review before responding

## Run locally

Install dependencies:

    pip install -r requirements.txt

Configure the GROQ_API_KEY secret, then run:

    streamlit run app.py

## Security

Never commit API keys, email credentials, or real customer complaints.

AI-generated resolutions require human review.

## Storage note

The current prototype uses local CSV and JSON files.
Use persistent database storage for production deployment.
