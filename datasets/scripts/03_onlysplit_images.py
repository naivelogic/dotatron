import os, sys
import argparse


if __name__ == '__main__':
    #sys.path.append("../DOTA_devkit/")
    sys.path.append("datasets/dota_devkit")
    from SplitOnlyImage_multi_process import splitbase

    args = argparse.Namespace()
    args.imagedir = "/home/redne/LUIA_challenge_dev/tron_dota/ws/dota_dataset/planes_example/images/"
    args.newimagedir = "/home/redne/LUIA_challenge_dev/tron_dota/ws/dota_dataset/planes_split5/"
    args.subsize = 1024
    args.gap = 600
    args.num_process = 16

    if not os.path.exists(args.newimagedir):
        os.makedirs(args.newimagedir)  # make new output folder

    split = splitbase(args.imagedir,
                      args.newimagedir,
                      subsize=args.subsize,
                      gap=args.gap,
                      num_process=args.num_process)
    split.splitdata(1)

    print("Spliting ImageOnly complete. New Dir: ", args.newimagedir)