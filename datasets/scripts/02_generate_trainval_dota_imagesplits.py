import os, sys
import argparse


if __name__ == '__main__':
    #sys.path.append("../")
    sys.path.append("datasets/dota_devkit")
    from ImgSplit_multi_process import splitbase, remove_no_detect_files

    args = argparse.Namespace()
    args.imagedir = "/home/redne/LUIA_challenge_dev/tron_dota/ws/dota_dataset/planes_example/"
    args.newimagedir = "/home/redne/LUIA_challenge_dev/tron_dota/ws/dota_dataset/planes_split4/train/"
    args.subsize = 1024
    args.gap = 200
    args.num_process = 16

    if not os.path.exists(args.newimagedir):
        os.makedirs(args.newimagedir)  # make new output folder

    # 1,830 original TRAIN images
    # after split 
    # 35,720,894,544 bytes (35GB) & 24,149 images
    # Time used: 1.53037
    split = splitbase(args.imagedir,
                        args.newimagedir,
                        gap=args.gap,        
                        subsize=args.subsize,   
                        num_process=args.num_process
                        )
    #split.labelpath = split.labelpath.replace('labelTxt','labelTxt-v2.0')
    split.splitdata(1)  # resize rate before cut

    remove_no_detect_files(args.newimagedir)

    print("Spliting Dataset complete. New Dir: ", args.newimagedir)