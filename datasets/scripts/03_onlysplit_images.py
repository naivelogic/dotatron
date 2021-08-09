import os, sys
import argparse


if __name__ == '__main__':
    sys.path.append("datasets/dota_devkit")
    from SplitOnlyImage_multi_process import splitbase

    args = argparse.Namespace()
    #args.imagedir = "/home/redne/LUIA_challenge_dev/tron_dota/ws/dota_dataset/planes_example/images/"
    #args.newimagedir = "/home/redne/LUIA_challenge_dev/tron_dota/ws/dota_dataset/planes_split5/"
    args.imagedir = "/mnt/omreast_users/phhale/open_ds/DOTA_aerial_images/images/test_challenge/part2_images/"
    args.newimagedir = "/mnt/omreast_users/phhale/open_ds/DOTA_aerial_images/images/test_challenge/part2_split1024v200/"
    args.subsize = 1024
    args.gap = 200 #600
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