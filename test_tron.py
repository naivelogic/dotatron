from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os
from tqdm import tqdm

from tools.dota2_defaults import DOTA2CLASSES
from tools.args_lib import test_tron_args
from tools.test_model import load_dotatron_model

import warnings
warnings.filterwarnings("ignore")


def get_image_list(input_dir, num_images=False):
    image_list = os.listdir(input_dir)

    if num_images != False:
        # randomly sample image list with num_images
        import random
        image_list = random.sample(image_list, num_images)

    return image_list

def visualize_prediction(outputs, args):
    # if there are no instances dont save image
    if len(outputs['instances'].to("cpu").get_fields()['scores']) < 1:
        #continue 
        return None 

    # Draw on the image
    # note switch to RGB order for visualizer
    v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("dota2"))
    if not args.use_gpu:
        v = v.draw_instance_predictions(outputs['instances'])
    else:
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))

    # Save the output
    output_image_filename = os.path.join(args.output_dir, image)
    os.makedirs(os.path.dirname(output_image_filename), exist_ok=True)
    cv2.imwrite(output_image_filename, v.get_image()[:, :, ::-1])

    


if __name__ == '__main__':
    args = test_tron_args().parse_args()
    print("Command Line Args:", args)

    # setup_output_folder
    model_ckp_name = os.path.basename(args.ckpt_path).split('_')[1][:-4] #['model', '0005999.pth'] #'0005999'
    
    args.output_dir = os.path.join(args.output_dir, model_ckp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    model_dir = os.path.dirname(args.ckpt_path) # experiments/detectron2/080421_dotatrainval_devkit_frcnn_X101_nc24x01
    model_name = os.path.basename(model_dir) # '080421_dotatrainval_devkit_frcnn_X101_nc24x01'
    args.config_filename = os.path.join(model_dir, 'config.yaml')
    predictor = load_dotatron_model(args)
    
    # Create labels for inference 
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get("dota2").thing_classes = DOTA2CLASSES

    image_list = get_image_list(args.input_dir, args.num_images)

    for image in tqdm(image_list):
        # Load the image and run it through the model
        input_image_filename = os.path.join(args.input_dir, image)
        im = cv2.imread(input_image_filename)
        im = im[:, :, ::-1]
        outputs = predictor(im)  # note BGR order for predictor input

        visualize_prediction(outputs, args)

    print(f">> predictions complete for {model_name}")
    print(f">> Results path: {args.output_dir}")
        

        