## Data Storyteller -- Data Visualization with Natural Language

A simple app to generate visualiations from the following ingredients:
- Data (tabular eg. csv)
- Prompt (user's intent; eg. 'show how x varies with y and z')
- Guidelines (optional; pre-embedded)

## To run
Run nicegui_static.py to launch nicegui app (it loads the ui in your browser) -- see [nicegui](https://nicegui.io/)

You can make a .env file with LLM API keys and a debug flag (see [.env.example](.env.example))

## Knowledge Base
The knowledge base of best practises is in [json_chart_types](json_chart_types.json) -- this follows the general format of mapping chart types to associated best practises, but it can be modified with suitable changes to the base langchain pipeline of the [Storyteller](DST_GPT_v5.py)
