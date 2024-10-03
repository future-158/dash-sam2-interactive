import json
from types import SimpleNamespace
from typing import List, Optional

import dash
import fiftyone as fo
import numpy as np
import plotly.express as px
import requests
import torch
from dash import Input, Output, ctx, dcc, html
from PIL import Image, ImageDraw, ImageOps
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

if fo.dataset_exists(ds_name := "sam2-interactive_test_dataset"):
    ds = fo.load_dataset(ds_name)
else:
    def load_image(url: str) -> Image.Image:
        if url.startswith("http://") or url.startswith("https://"):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"
            }
            response = requests.get(url, stream=True, headers=headers)
            if response.status_code == 200:
                image = Image.open(response.raw)
            else:
                raise ValueError(
                    f"Failed to fetch image from URL. HTTP status code: {response.status_code}"
                )

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image

    img = load_image(
        "https://images.pexels.com/photos/60050/huskies-husky-blue-eye-dog-60050.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    )
    img.thumbnail((1280, 720))
    img.save("artifact/dogs.jpg")

    img = load_image(
        "https://images.pexels.com/photos/3749012/pexels-photo-3749012.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    )
    img.thumbnail((1280, 720))
    img.save("artifact/wolves.jpg")

    ds = fo.Dataset()
    ds.add_sample(
        fo.Sample(
            filepath="artifact/dogs.jpg",
            detections=fo.Detections(),
        )
    )
    ds.add_sample(
        fo.Sample(
            filepath="artifact/wolves.jpg",
            detections=fo.Detections(),
        )
    )
    ds.compute_metadata()
    ds.name = ds_name
    ds.persistent = True


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


def draw_points(
    img: Image.Image, pos_points: List[tuple], neg_points: List[tuple], radius: int = 3
):
    """
    Draw a point on an image
    """
    draw = ImageDraw.Draw(img)
    for center_x, center_y in pos_points:
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill="red",
        )

    for center_x, center_y in neg_points:
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill="green",
        )
    return img




def box_contain_point(box, x, y):
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def pil_to_fig(img: Image.Image):
    fig = px.imshow(img)
    return fig.update_layout(
        xaxis=dict(
            range=[0, img.width],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[0, img.height],
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="select",  # Enable selection
        clickmode="event+select",
    )




# manage global state
st = SimpleNamespace()
st.session_state = SimpleNamespace()
st.session_state.sample_id = ds.first().id
st.session_state.toggle_box = "selectButton"

st.session_state.pos_points = [[] for i in range(50)]
st.session_state.neg_points = [[] for i in range(50)]
st.session_state.bboxes = []
st.session_state.focused = None

sample = ds[st.session_state.sample_id]
img = Image.open(sample.filepath).convert("RGB")
predictor.set_image(np.array(img))



# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Bounding Box Manager"

initial_fig = pil_to_fig(img)
# Define the app layout
app.layout = html.Div(
    [
        dcc.Dropdown(ds.values("id"), ds.first().id, id="sample-id-dropdown"),
        html.Button("Inference", id="infButton"),
        html.Button("Select", id="selectButton"),
        html.Button("Positive", id="posButton"),
        html.Button("Negative", id="negButton"),
        html.Div(id="output"),
        dcc.Graph(
            id="image-graph",
            figure=initial_fig,
            config={
                "staticPlot": False,  # Allows user interactions
                "scrollZoom": False,
                "doubleClick": "reset",  # Reset on double-click
                "displayModeBar": True,  # Show the mode bar for drawing tools
                "modeBarButtonsToAdd": [
                    "drawrect",
                    "eraseshape",
                ],  # Add rectangle and erase buttons
                "modeBarButtonsToRemove": ["toImage", "zoom2d", "pan2d", "select2d"],
            },
            style={"width": "1000px", "height": "1000px"},
        ),
        dcc.Graph(
            id="seg-graph",
            figure=pil_to_fig(Image.new("RGB", size=img.size)),
            style={"width": "500px", "height": "500px"},
        ),
    ]
)


app.clientside_callback(
    """
        function(id) {
            document.addEventListener("keydown", function(event) {
                if ( true) {
                    if (event.key == 'q') {
                        document.getElementById('selectButton').click()
                        event.stopPropogation()
                    }
                    if (event.key == 'w') {
                        document.getElementById('posButton').click()
                        event.stopPropogation()
                    }
                    if (event.key == 'e') {
                        document.getElementById('negButton').click()
                        event.stopPropogation()
                    }
                }
            });
            return window.dash_clientside.no_update       
        }
    """,
    Output("selectButton", "id"),
    Input("selectButton", "id"),
)


@app.callback(
    Output("image-graph", "figure", allow_duplicate=True),
    Output("seg-graph", "figure", allow_duplicate=True),
    Input("sample-id-dropdown", "value"),
    prevent_initial_call=True,
)
def update_output(value):
    st.session_state.sample_id = value
    st.session_state.toggle_box = "selectButton"

    st.session_state.pos_points = [[] for i in range(50)]
    st.session_state.neg_points = [[] for i in range(50)]
    st.session_state.bboxes = []
    st.session_state.focused = None

    sample = ds[st.session_state.sample_id]
    img = Image.open(sample.filepath)
    predictor.set_image(np.array(img))
    print("image set")    

    return pil_to_fig(img), pil_to_fig(Image.new("RGB", size=img.size))


@app.callback(
    Output("output", "children"),
    Input("selectButton", "n_clicks"),
    Input("posButton", "n_clicks"),
    Input("negButton", "n_clicks"),
    prevent_initial_call=True,
)
def show_value(n1, n2, n3):
    st.session_state.toggle_box = ctx.triggered_id
    print(st.session_state.toggle_box)
    return ctx.triggered_id


@app.callback(
    Output("seg-graph", "figure"),
    Input("infButton", "n_clicks"),
    prevent_initial_call=True,
)
def inference_image(n_clicks):
    # raise dash.exceptions.PreventUpdate
    if n_clicks < 1:
        raise dash.exceptions.PreventUpdate

    sample = ds[st.session_state.sample_id]
    img = Image.open(sample.filepath)
    merge_mask = np.zeros(img.size[::-1], dtype=bool)
    for box, pos_points, neg_points in zip(
        st.session_state.bboxes,
        st.session_state.pos_points,
        st.session_state.neg_points,
    ):

        point_coords = [*pos_points, *neg_points]
        point_labels = [1] * len(pos_points) + [0] * len(neg_points)

        if not point_coords:
            continue

        masks, _, _ = predictor.predict(
            box=box,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        assert merge_mask.shape == masks[0].shape
        merge_mask |= masks[0] > 0
    
    pil_mask = Image.fromarray(merge_mask).convert("RGB")
    return pil_to_fig(pil_mask)


@app.callback(
    Output("image-graph", "figure", allow_duplicate=True),
    Input("image-graph", "clickData"),
    prevent_initial_call=True,
)
def display_click_data(clickData):
    
    """
    {
    "points": [
        {
        "curveNumber": 0,
        "x": 84,
        "y": 23,
        "colormodel": "rgb",
        "bbox": {
            "x0": 91.5,
            "x1": 92.5,
            "y0": 344,
            "y1": 344
        }
        }
    ]
    }
    """
    if not clickData:
        raise dash.exceptions.PreventUpdate
    if "points" not in clickData:
        raise dash.exceptions.PreventUpdate
    

    x = clickData["points"][0]["x"]
    y = clickData["points"][0]["y"]
    sample = ds[st.session_state.sample_id]

    match st.session_state.toggle_box:
        case "selectButton":        
            candidates  = [id for id, box in enumerate(st.session_state.bboxes) if box_contain_point(box, x, y)]
            if not candidates:
                return dash.no_update
            
            st.session_state.focused = candidates[0]             
            img = Image.open(sample.filepath)
            fig = draw_bboxes(img, st.session_state.bboxes, st.session_state.focused)            
            return fig
        
        case "posButton":
            nth = st.session_state.focused
            if nth is not None and  box_contain_point(st.session_state.bboxes[nth], x, y):
                st.session_state.pos_points[nth].append((x, y))
                img = Image.open(sample.filepath)
                fig = draw_bboxes(img, st.session_state.bboxes, nth)                        
                return fig

        case "negButton":
            nth = st.session_state.focused
            if nth is not None and   box_contain_point(st.session_state.bboxes[nth], x, y):
                st.session_state.neg_points[nth].append((x, y))
                img = Image.open(sample.filepath)
                fig = draw_bboxes(img, st.session_state.bboxes, nth)                        
                return fig

        case _:
            raise ValueError(f"Unknown toggle_box: {st.session_state.toggle_box}")

    


@app.callback(
    Output("image-graph", "figure", allow_duplicate=True),
    Input("image-graph", "selectedData"),
    prevent_initial_call=True,
)
def display_selected_data(selectedData):
    """
    {
    "points": [],
    "range": {
        "x": [
        73,
        276
        ],
        "y": [
        6.328125,
        218.328125
        ]
    }
    }
    """

    if not selectedData:
        raise dash.exceptions.PreventUpdate
    if "range" not in selectedData:
        raise dash.exceptions.PreventUpdate

    sample = ds[st.session_state.sample_id]
    img = Image.open(sample.filepath)
    x0, x1 = selectedData["range"]["x"]
    y0, y1 = selectedData["range"]["y"]

    assert x0 < x1
    assert y0 < y1

    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(img.width, int(x1))
    y1 = min(img.height, int(y1))

    bbox = [x0, y0, x1, y1]
    st.session_state.bboxes.append(bbox)
    st.session_state.focused = len(st.session_state.bboxes) - 1
    # mask[len(bboxes)-1, y0:y1, x0:x1, ] = True
    fig = draw_bboxes(img, st.session_state.bboxes)
    return fig


def draw_bboxes(img: Image.Image, bboxes: List, focus_id: Optional[int] = None):
    draw = ImageDraw.Draw(img)
    if focus_id is None:
        focus_id = len(bboxes) - 1
    for i, bbox in enumerate(bboxes):
        if i == focus_id:
            color = "red"
        else:
            color = "green"
        draw.rectangle(bbox, outline=color, width=2)
        if i == focus_id:
            pos_points = st.session_state.pos_points[i]
            neg_points = st.session_state.neg_points[i]
            draw_points(img, pos_points, neg_points)

    fig = pil_to_fig(img)
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False, port=8050)
