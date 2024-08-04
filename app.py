import base64
import plotly.express as px
import cv2
from pathlib import Path
from PIL import ImageDraw
from uuid import uuid4
from io import BytesIO
from PIL import Image
import json
import dash
import numpy as np
from dash import Input, Output, State, dcc, html
from diffusers.utils import load_image
from types import SimpleNamespace
import random
import torch
import os
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


sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

def update():
    urls = [
        "https://images.pexels.com/photos/1806771/pexels-photo-1806771.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        "https://images.pexels.com/photos/8919231/pexels-photo-8919231.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/3044506/pexels-photo-3044506.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/2349891/pexels-photo-2349891.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/17860408/pexels-photo-17860408/free-photo-of-rider-rearing-up-his-horse.jpeg?auto=compress&cs=tinysrgb&w=800",
    ]
    url = random.choice(urls)
    img = load_image(url)
    img.thumbnail((1280, 720))

    st.session_state["img"] = img
    st.session_state["pos"] = []
    st.session_state["neg"] = []
    

update()


def pil_to_fig(image):
    fig = px.imshow(image)
    fig.update_layout(
        {
            "xaxis": {"showgrid": False, "showticklabels": False},
            "yaxis": {"showgrid": False, "showticklabels": False},
        }
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # fig.update_layout(dragmode='drawrect')
    fig.update_layout(dragmode="select")
    return fig


def draw_points():
    pos = st.session_state["pos"]
    neg = st.session_state["neg"]

    xs = [x for x, y in pos + neg]
    ys = [y for x, y in pos + neg]
    labels = ["pos"] * len(st.session_state["pos"]) + ["neg"] * len(
        st.session_state["neg"]
    )

    img = st.session_state["img"].copy()
    draw = ImageDraw.Draw(img)
    
    def draw_circle(draw, center, radius, color):
        x, y = center
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], fill=color, outline=color)

    for x, y, label in zip(xs, ys, labels):
        if label == "pos":
            draw_circle(draw, (x, y), 10, "green")
        else:
            draw_circle(draw, (x, y), 10, "red")
    return img    


# Initialize the Dash app
app = dash.Dash(__name__)


app.layout = html.Div(
    [
        html.Button(id="button-update", n_clicks=0, children="Update Image"),
        html.Button(id="button-save", n_clicks=0, children="Save Image and Mask"),
        dcc.RadioItems(
            id="radio",
            options=[
                {"label": "pos", "value": "pos"},
                {"label": "neg", "value": "neg"},
            ],
            value="pos",
        ),
        dcc.Graph(id="graph-active", figure=pil_to_fig(st.session_state["img"])),
    ]
)


@app.callback(
    Output("graph-active", "figure", allow_duplicate=True),
    Input("button-update", "n_clicks"),
    prevent_initial_call=True,
)
def update_image(n_clicks):
    if n_clicks > 0:
        update()
        return pil_to_fig(st.session_state["img"])
    return dash.no_update


@app.callback(
    Input("button-save", "n_clicks"),
    prevent_initial_call=True,
)
def save_image(n_clicks):
    if n_clicks > 0:
        
        out_dir = Path("./artifact")
        out_dir.mkdir(parents=True, exist_ok=True)

        dest = (out_dir / uuid4().hex).with_suffix(".png")
        img = Image.new(
            "RGB",
            (st.session_state["img"].width * 2, st.session_state["img"].height * 2),
            (255, 255, 255),
        )
        img.paste(
            st.session_state["img"],
            (0, 0),
        )
        img.paste(
            st.session_state["mask_image"],
            (st.session_state["img"].width, 0),
        )
        img.save(dest)

    return dash.no_update


@app.callback(
    Output("graph-active", "figure", allow_duplicate=True),
    Input("graph-active", "clickData"),
    State("radio", "value"),
    prevent_initial_call=True,
)
def display_click_data(clickData,  radio):
    if "points" not in clickData:
        raise dash.exceptions.PreventUpdate

    y = clickData["points"][0]["y"]
    x = clickData["points"][0]["x"]

    if radio == "pos":
        st.session_state["pos"].append((x, y))
    else:
        st.session_state["neg"].append((x, y))

    annotated_img = draw_points()
    return pil_to_fig(annotated_img)


@app.callback(
    Output("graph-active", "figure", allow_duplicate=True),
    Input("graph-active", "relayoutData"),
    prevent_initial_call=True,
)
def update_graph(relayoutData,):

    # relayoutData: {'selections': [{'xref': 'x', 'yref': 'y', 'line': {'width': 1, 'dash': 'dot'}, 'type': 'rect', 'x0': 805.9000000000001, 'y0': 679.5, 'x1': 186.70000000000005, 'y1': 116.3}]}
    # selectedData: {'points': [], 'range': {'x': [244.30000000000004, 834.7000000000002], 'y': [175.5, 677.9]}}

    if "selections" in relayoutData:
        selection = relayoutData["selections"][0]
        
        x_start = selection["x0"]
        y_start = selection["y0"]
        x_end = selection["x1"]
        y_end = selection["y1"]

        x0, x1 = sorted([x_start, x_end])
        y0, y1 = sorted([y_start, y_end])

        if x_start < x_end and y_start < y_end:
            # normal bbox selection

            pos = st.session_state["pos"]
            neg = st.session_state["neg"]

            points = pos + neg
            labels = [1] * len(st.session_state["pos"]) + [0] * len(st.session_state["neg"])

            input_box = np.array([x0, y0, x1, y1])
            input_point = np.array(points) if len(points) else None
            input_label = np.array(labels) if len(labels) else None


            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(np.array(st.session_state["img"]))
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=False,
                )

            mask = masks[0] > 0        
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            mask = np.where(mask[...,None], color, 0)
            mask = (mask * 255).astype(np.uint8)
            st.session_state["mask_image"] = Image.fromarray(mask)
            new_img = Image.alpha_composite(
                st.session_state["img"].copy().convert("RGBA"),
                st.session_state["mask_image"],
                ).convert("RGB")
            return pil_to_fig(new_img)
        
        elif x_start < x_end and y_start > y_end:
            # advanced usage. it add positive point in start point (bottom left corner)
            st.session_state["pos"].append((x_start, y_start))    
            annotated_img = draw_points()
            return pil_to_fig(annotated_img)
        
        elif x_start > x_end and y_start > y_end:
            # advanced usage. it add negative point in start point (bottom left corner)
            st.session_state["neg"].append((x_start, y_start))
            annotated_img = draw_points()
            return pil_to_fig(annotated_img)            
    else:
        raise dash.exceptions.PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
