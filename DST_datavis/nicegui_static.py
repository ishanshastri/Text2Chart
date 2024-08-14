#!/usr/bin/env python3
import io
from nicegui import run, ui
from nicegui.events import UploadEventArguments
from DST_GPT_v5 import StoryTeller, DEFAULT_LLM_CONFIGS, LLM_ENDPOINTS
from DST_rag_utils_v2 import *
from typing import Optional, List

JSON_GUIDELINE_FNAME = '/Users/ishanshastri/Desktop/nus/DST/main_proj/json_chart_types.json'#'json_chart_types.json'

default_llm_configs = DEFAULT_LLM_CONFIGS.copy()
llm_endpoints = LLM_ENDPOINTS.copy()

retriever = get_langchain_retriever_from_json_file(fname=JSON_GUIDELINE_FNAME, search_kwargs={'k':2}, max_chunk_size=30)

storyteller = StoryTeller(llm_name="huggingface", retriever=retriever)

prompt_params = storyteller.get_default_params()

ordered_fallback_models_dict = {
    'huggingface':default_llm_configs['huggingface'],
    'google':default_llm_configs['google'],
    'openai':default_llm_configs['openai']
}

ordered_fallback_models = ['huggingface', 'google', 'openai']

async def upload_csv(e: UploadEventArguments):
    csv = e.content.read().decode('utf-8').splitlines(keepends=True)
    print(",".join(csv))
    global storyteller, prompt_params
    storyteller.set_df_from_csv_str(",".join(csv))
    print("csv uploaded")

async def generate_image():
    print("model being used to generate (in nicegui): ", storyteller.llm)
    image = ui.image().style('width: 30em')
    image.source = 'https://dummyimage.com/600x400/ccc/000000.png&text=building+image...'
    prediction=None

    # Code generation
    try:
        '''Generate with code-fixer llm fallback'''
        # prediction = await run.io_bound(storyteller.generate_with_fallback, 
        #                                 prompt=prompt.value, 
        #                                 library='plotly',
        #                                 get_trace_url=True)
        
        '''Generate with LLM fallback'''
        prediction = await run.io_bound(storyteller.generate_with_llm_fallback, 
                                        prompt=prompt.value, 
                                        library='plotly',
                                        get_trace_url=True,
                                        llms_and_configs=ordered_fallback_models_dict)
        
        print("url: ", prediction.get('url', "nothing"))

    except Exception as e:
        print("error during call to generate: ", str(e))
        ui.label(f"ERROR: {str(e)}")

    image.delete()
    if prediction and prediction.get('fig', None):
        ui.plotly(prediction['fig'])
        lm = prediction.get('llm', "unsure.")
        ui.label(f'Following LLM was used: {lm}')

async def toggle_guideline_selection():
    if not guidelines_txtbox.value:
        return
    if switch.value == True:
        global retriever
        s = guidelines_txtbox.value
        dummy_ret = DummyRetriever(documents=guidelines_txtbox.value.split('\n'))
        storyteller.set_main_retriever(retriever=dummy_ret)

    else:
        storyteller.set_main_retriever(retriever=retriever)
    
async def update_storyteller(endpoint_selected='huggingface'):
    print("model updated to: ", endpoint_selected)
    global ordered_fallback_models_dict, retriever
    new_key_list = [endpoint_selected]+list(ordered_fallback_models_dict.keys())
    new_ordered_fallback_models_dict = dict.fromkeys(new_key_list)
    new_ordered_fallback_models_dict = {k: ordered_fallback_models_dict[k] for k in new_key_list}
    ordered_fallback_models_dict = new_ordered_fallback_models_dict

    storyteller.set_main_llm(endpoint_type=llm_endpoints[endpoint_selected], llm_config=default_llm_configs[endpoint_selected])
    storyteller.set_generator_prompts(step_gen_prompt=prompt_params['Step Retrieval Template'],
                                      code_gen_prompt=prompt_params['Code Generation Template'])

with ui.row().style('gap:10em'):
    # CSV upload and prompt input
    with ui.column():
        ui.label('CSV Upload').classes('text-2xl')
        ui.upload(on_upload=upload_csv, auto_upload=True).style('width: 20em')
        transcription = ui.label().classes('text-xl')

        ui.label('Chart').classes('text-2xl')
        prompt = ui.input('prompt').style('width: 40em')
        ui.button('Generate', on_click=generate_image).style('width: 15em')

    # Prompt and LLM params editboxes
    parameters = ['question', 'something']
    with ui.column():
        switch = ui.switch('Use Custom Guidelines (Override Knowledge Base)', on_change=toggle_guideline_selection)
        guidelines_txtbox = ui.textarea(label="Guidelines").style('width: 30em').bind_visibility_from(switch, 'value')
        set_guidelines_button = ui.button('Update', on_click=toggle_guideline_selection).style('width: 5em').bind_visibility_from(switch, 'value')
        ui.label('Prompt Parameters').classes('text-2xl')
        for k, v in prompt_params.items():
            with ui.row():
                ui.textarea(label=k, value=v).style('width: 30em').bind_value(prompt_params, target_name=k)
        ui.label('LLM Parameters').classes('text-2xl')
        with ui.tabs() as tabs:
            ui.tab('huggingface', label='Huggingface')
            ui.tab('google', label='Google')
            ui.tab('openai', label='OpenAI')
        with ui.tab_panels(tabs, value='huggingface').classes('w-full'):
            with ui.tab_panel('huggingface'):
                for k, v in default_llm_configs['huggingface'].items():
                    with ui.row():
                        ui.input(label=k, value=v).bind_value(ordered_fallback_models_dict['huggingface'], target_name=k)
            with ui.tab_panel('google'):
                ui.label('Google')
                for k, v in default_llm_configs['google'].items():
                    with ui.row():
                        ui.input(label=k, value=v).bind_value(ordered_fallback_models_dict['google'], target_name=k)

            with ui.tab_panel('openai'):
                ui.label('OpenAI')
                for k, v in default_llm_configs['openai'].items():
                    with ui.row():
                        ui.input(label=k, value=v).bind_value(ordered_fallback_models_dict['openai'], target_name=k)

        ui.button('Update', on_click=lambda : update_storyteller(tabs.value)).style('width: 15em')
                
ui.run()
