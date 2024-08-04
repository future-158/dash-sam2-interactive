import base64
import plotly.express as px
import cv2
from sklearn.preprocessing import minmax_scale
import pandas as pd
from io import BytesIO
from PIL import Image
import json
import pandas as pd
from io import BytesIO

import dash
import fiftyone as fo
import numpy as np
import plotly.express as px
from dash import Input, Output, State, dcc, html
from PIL import Image
from diffusers.utils import load_image
from types import SimpleNamespace
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['TORCH_CUDNN_SDPA_ENABLED']='1'

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor




st = SimpleNamespace()
st.session_state = {}

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "/workspace/provision/models/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
# st.session_state['predictor'] = predictor



def update():
    urls = [
        'https://images.pexels.com/photos/1806771/pexels-photo-1806771.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
        'https://images.pexels.com/photos/1806771/pexels-photo-1806771.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
        'https://images.pexels.com/photos/1806771/pexels-photo-1806771.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    ]
    url = random.choice(urls)
    img = load_image(url)
    img.thumbnail((1280, 720))    
    # st.session_state['predictor'].set_image(np.array(img))
    st.session_state['img'] = img
    st.session_state['pos'] = []
    st.session_state['neg'] = []

update()

def pil_to_fig(image):
    fig = px.imshow(image)
    fig.update_layout(
        {
            'xaxis': {'showgrid': False,
                      'showticklabels': False
                      },
            'yaxis': {'showgrid': False,
                      'showticklabels': False
                      }
        }

    )
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.update_layout(dragmode='drawrect')


# todo: update image button
# todo: save image button

def fig_to_pil(figure_data) -> Image.Image:
    # Decode plotly figure to an pillow image

    if  figure_data["data"][0]['type'] == 'heatmap':
        return Image.fromarray(np.array(figure_data["data"][0]['z']).astype(np.uint8))

    url = figure_data["data"][0]["source"]    
    encoded_image = url.split(";base64,")[-1]
    return Image.open(BytesIO(base64.b64decode(encoded_image)))    

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.RadioItems(
        id = 'radio',
        options=[
            {'label': 'pos', 'value': 'pos'},
            {'label': 'neg', 'value': 'neg'},
        ],
        value='pos',
    ),
    dcc.Graph(id='graph-active', figure=pil_to_fig(st.session_state['img'])),
    # dcc.Graph(id='graph-responsive', figure=pil_to_fig(img_responsive)),
    # dcc.Graph(id='graph-mask', figure=pil_to_fig(mask_image)),
    html.Pre(id='click')
])

@app.callback(
    Output('click', 'children', allow_duplicate=True),
    Output('graph-active', 'figure', allow_duplicate=True),
    Input('graph-active', 'clickData'),
    State('graph-active', 'figure'),
    State('radio', 'value'),
    # State('graph-mask', 'figure'),
    # State('graph-responsive', 'figure'),
    prevent_initial_call=True   
)
def display_click_data(clickData, figure_active, radio):

    if 'points' not in clickData:
        raise dash.exceptions.PreventUpdate
    
    # img_active = fig_to_pil(figure_active)
    y = clickData['points'][0]['y']
    x = clickData['points'][0]['x']

    if radio == 'pos':
        st.session_state['pos'].append((x, y))
    else:
        st.session_state['neg'].append((x, y))


    pos = st.session_state['pos']
    neg = st.session_state['neg']

    xs = [x for x, y in pos + neg]
    ys = [y for x, y in pos + neg]
    labels = ['pos']*len(st.session_state['pos']) + ['neg']*len(st.session_state['neg'])

    img = st.session_state['img'].copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    def draw_circle(draw, center, radius, color):
        x, y = center
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], fill=color, outline=color)    

    for x,y,label in zip(xs, ys, labels):
        if label == 'pos':
            draw_circle(draw, (x, y), 10, 'green')
        else:
            draw_circle(draw, (x, y), 10, 'red')        
    return json.dumps(clickData, indent=2), pil_to_fig(img)


@app.callback(
    Output('graph-active', 'figure', allow_duplicate=True),
    Input('graph-active', 'relayoutData'),
    State('graph-active', 'figure'),
    prevent_initial_call=True   
)
def update_graph(relayout_data, figure_data):
    if "shapes" in relayout_data:
        shape = relayout_data["shapes"][-1]
        # print(shape)

        x0, y0, x1, y1 = map(int, (shape["x0"], shape["y0"], shape["x1"], shape["y1"]))

    
        pos = st.session_state['pos']
        neg = st.session_state['neg']

        points = pos + neg
        labels = [1]*len(st.session_state['pos']) + [0]*len(st.session_state['neg'])


        input_box = np.array([x0, y0, x1, y1])
        input_point = np.array(points) if len(points) else None
        input_label = np.array(labels) if len(labels) else None

        # print(input_box, input_point, input_label)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(np.array(st.session_state['img']))
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
            )
            
        mask = masks[0] > 0    
        color = np.array([30/255, 144/255, 255/255, 0.6])

        h, w = mask.shape
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        mask_image = (mask_image * 255).astype(np.uint8)
        st.session_state['mask_image'] = mask_image
        
        seg_img = Image.fromarray(mask_image)
        img = st.session_state['img'].copy()
        new_img = Image.alpha_composite(img.convert('RGBA'), seg_img).convert('RGB')
        # img.paste((255, 0, 0), (x0, y0, x1, y1))
        return px.imshow(new_img).update_layout(dragmode='drawrect')
    else:
        raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)









