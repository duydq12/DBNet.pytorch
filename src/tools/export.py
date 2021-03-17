import onnx
import torch

from src.models import build_model

device = torch.device("cpu")


def export_static_input():
    checkpoint = torch.load("/home/rabiloo/project/alpr/DBNet.pytorch/weights/model_best.pth",
                            map_location=device)

    config = checkpoint['config']
    config['arch']['backbone']['pretrained'] = False
    db_model = build_model(config['arch'])
    db_model.load_state_dict(checkpoint['state_dict'])
    db_model.to(device)
    db_model.eval()

    input_name = ['input']
    output_name = ['output']

    input = torch.randn(1, 3, 640, 640)

    torch.onnx.export(db_model, input,
                      "/home/rabiloo/project/alpr/DBNet.pytorch/weights/model_best.onnx",
                      input_names=input_name,
                      output_names=output_name,
                      verbose=True,
                      opset_version=11,
                      export_params=True,
                      keep_initializers_as_inputs=True,
                      # dynamic_axes={"input": {3: "time_step"},
                      #               "output": {3: "time_step"}}
                      )

    print('export done')


def export_dynamic_input():
    model = onnx.load('/home/rabiloo/project/alpr/DBNet.pytorch/weights/model_best.onnx')
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
    onnx.save(model, '/home/rabiloo/project/alpr/DBNet.pytorch/weights/model_best_dynamic.onnx')

    print('export done')


export_dynamic_input()
