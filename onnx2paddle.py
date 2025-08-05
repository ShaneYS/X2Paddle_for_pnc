import x2paddle
from x2paddle import convert
from pathlib import Path
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import onnxruntime as ort
import paddle


def convert():
    save_dir = "/data/project/ys_code/X2Paddle_for_pnc/test_result/202508011228"
    onnx_path = "/data/project/ys_code/X2Paddle_for_pnc/test_result/202508011228/all_scene_prediction_model_202508011228_sim.onnx"
    onnx_export_log_path = "/data/project/ys_code/X2Paddle_for_pnc/test_result/202508011228/export_202508011228.log"

    # onnx to paddle
    batch_size = 1
    input_shape_dict = {
        "obstacle_polyline": [batch_size, 64, 16, 17],
        "lane_polyline": [batch_size, 256, 64, 20],
        "stopline_polyline": [batch_size, 16, 8, 16],
        "crosswalk_polyline": [batch_size, 32, 16, 5],
        "adc_occ_map": [batch_size, 400, 400, 2],
        "veh_target_obs_idx": [16],
        "cyc_target_obs_idx": [10],
        "ped_target_obs_idx": [10],
        "anchor_lane_target_obs_idx": [4],
        "scene_index": [1],
        "anchor_lane_indices": [4, 8],
        "veh_target_obs_idx_mask_adc_flag": [16],
        "cyc_target_obs_idx_mask_adc_flag": [10],
        "ped_target_obs_idx_mask_adc_flag": [10],
        "anchor_lane_target_obs_idx_mask_adc_flag": [4],
    }
    print(f"onnx_path={onnx_path}")
    x2paddle.convert.onnx2paddle(
        model_path=onnx_path,
        save_dir=save_dir,
        input_shape_dict=str(input_shape_dict),
        enable_optim=True,
        onnx_export_log_path=onnx_export_log_path,
    )

    print("convert done")


def load():
    # load pdmodel
    model = paddle.jit.load(
        "/data/project/ys_code/X2Paddle_for_pnc/test_result/202508011228/inference_model/model"
    )
    model.eval()
    print(model)
    
    paddle.summary(
        model,
        input_size=[
            (1, 64, 16, 17),
            (1, 256, 64, 20),
            (1, 16, 8, 16),
            (1, 32, 16, 5),
            (1, 400, 400, 2),
            (16),
            (10),
            (10),
            (4),
            (1),
            (4, 8),
            (16),
            (10),
            (10),
            (4),
        ],
    )
    
    # # 运行模型并捕获中间输出，定位错误层
    # with paddle.no_grad():
    #     try:
    #         valid_inputs = [
    #             paddle.randn([1, 64, 16, 17], dtype='float32'), 
    #             paddle.randn([1, 256, 64, 20], dtype='float32'),
    #             paddle.randn([1, 16, 8, 16], dtype='float32'),  
    #             paddle.randn([1, 32, 16, 5], dtype='float32'),  
    #             paddle.randn([1, 400, 400, 2], dtype='float32'),
    #             paddle.randn([16], dtype='float32'),
    #             paddle.randn([10], dtype='float32'),
    #             paddle.randn([10], dtype='float32'),
    #             paddle.randn([4], dtype='float32'),
    #             paddle.randn([1], dtype='float32'),
    #             paddle.randn([4, 8], dtype='float32'),
    #             paddle.randn([16], dtype='float32'),
    #             paddle.randn([10], dtype='float32'),
    #             paddle.randn([10], dtype='float32'),
    #             paddle.randn([4], dtype='float32')
                
    #         ]
    #         outputs = model(*valid_inputs)
    #         print(outputs)
    #     except Exception as e:
    #         print("错误发生在模型前向传播中：", e)
    #         # 可通过添加断点或打印中间变量逐步排查


    # # 打印所有层和参数信息
    # for name, param in model.named_parameters():
    #     print(f"层名称: {name}, 参数形状: {param.shape}")

    # # 打印子模块结构
    # for name, submodule in model.named_children():
    #     print(f"子模块: {name}, 类型: {type(submodule)}")
        
    pass

if __name__ == "__main__":
    # convert()
    load()
