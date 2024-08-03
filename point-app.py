import base64
import plotly.express as px
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

# ds = fo.load_dataset('magna-style-real-caption')



from types import SimpleNamespace
st = SimpleNamespace()
st.session_state = {}


def update():
    img = load_image('https://images.pexels.com/photos/1806771/pexels-photo-1806771.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2')
    img.thumbnail((1280, 720))    
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
    return fig


# todo: update image button
# todo: save image button


fig = st.session_state['img']


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

# input_point = np.array([[500, 375]])
# input_label = np.array([1])



# radio button



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
        value='MTL',
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


if __name__ == '__main__':
    app.run_server(debug=True, port=8052)