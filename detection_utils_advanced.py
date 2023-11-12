import subprocess


def run_detection():
    subprocess.call(
        #"python Detection_Advanced.py configs/det/faster_rcnn_r101_fpn_3x_det_bdd100k.py checkpoints/faster_rcnn_r101_fpn_3x_det_bdd100k.pth",
        "python Detection_Advanced.py configs/det/centernet_resnet18_dcnv2_140e_bdd100k.py checkpoints/centernet_resnet18_dcnv2_140e_bdd100k.pth",
        cwd=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/object_detection",
    )


# def run_detection():
#     subprocess.call(
#         "conda run -n bdd100k-mmdet python Detection.py configs/det/faster_rcnn_r101_fpn_3x_det_bdd100k.py checkpoints/faster_rcnn_r101_fpn_3x_det_bdd100k.pth",
#         cwd=r"C:/Users/grabe/Documents/Rayen/light_distribution_optimization/object_detection",
#         shell=True,
#     )
