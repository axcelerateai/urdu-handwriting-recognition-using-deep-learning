import csv
import os
from utils.data_utils import load_gt
from utils.accuracy_metrics import Levenshtein

def load_data(analyze_path):
    img_dir = os.path.join(analyze_path, "images")
    gt_dir = os.path.join(analyze_path, "labels/gt_char.csv")
    gt, filenames = load_gt(gt_dir, img_dir)
    
    return gt, filenames

def generate_accuracy_bins(filenames, gt, model, save_file):
    b_100 = []
    b_90 = []
    b_80 = []
    b_70 = []
    b_60 = []
    b_50 = []
    b_40 = []
    b_30 = []
    b_20 = []
    b_10 = []
    b_0 = []

    batch_size = 32
    for i in range(0, len(filenames), batch_size):
        f_pred, _ = model(filenames[i:i+batch_size])
        for j, f in enumerate(filenames[i:i+batch_size]): 
            f_acc = Levenshtein(labels=gt[i+j], logits=f_pred[j])

            if f_acc == 100:
                b_100.append(f)
            elif f_acc >= 90:
                b_90.append(f)
            elif f_acc >= 80:
                b_80.append(f)
            elif f_acc >= 70:
                b_70.append(f)
            elif f_acc >= 60:
                b_60.append(f)
            elif f_acc >= 50:
                b_50.append(f)
            elif f_acc >= 40:
                b_40.append(f)
            elif f_acc >= 30:
                b_30.append(f)
            elif f_acc >= 20:
                b_20.append(f)
            elif f_acc >= 10:
                b_10.append(f) 
            else:
                b_0.append(f)
        
        print("Done ", i, " images")

    w_b_100 = get_writers_in_bin(b_100)
    w_b_90 = get_writers_in_bin(b_90)
    w_b_80 = get_writers_in_bin(b_80)
    w_b_70 = get_writers_in_bin(b_70)
    w_b_60 = get_writers_in_bin(b_60)
    w_b_50 = get_writers_in_bin(b_50)
    w_b_40 = get_writers_in_bin(b_40)
    w_b_30 = get_writers_in_bin(b_30)
    w_b_20 = get_writers_in_bin(b_20)
    w_b_10 = get_writers_in_bin(b_10)
    w_b_0 = get_writers_in_bin(b_0)
   
    store_results(save_file, [b_100, b_90, b_80, b_70, b_60, b_50, b_40, b_30, b_20, b_10, b_0], 
            [w_b_100, w_b_90, w_b_80, w_b_70, w_b_60, w_b_50, w_b_40, w_b_30, w_b_20, w_b_10, w_b_0])

def get_writers_in_bin(b):
    # Assume each element in b is of format: */(writer_id)_*
    distinct_writers = []
    for f in b:
        f_name = f.split('/')[-1].split('.')[0]
        writer_id = int(f_name.split('_')[0])

        if not writer_id in distinct_writers:
            distinct_writers.append(writer_id)
    
    return distinct_writers

def store_results(save_file, a, b):
    if os.path.isfile(save_file):
        os.remove(save_file)

    with open(save_file, mode='w') as f:
        f_w = csv.writer(f, delimiter=',')
        f_w.writerow(['Accuracy','No. of Samples','No. of Distinct Writers','Writer IDs'])
        f_w.writerow(['100 ', str(len(a[0])), str(len(b[0])), ["%03d" % wid for wid in b[0]]])
        f_w.writerow(['90-100', str(len(a[1])), str(len(b[1])),["%03d" % wid for wid in b[1]]])
        f_w.writerow(['80-90', str(len(a[2])), str(len(b[2])), ["%03d" % wid for wid in b[2]]])
        f_w.writerow(['70-80', str(len(a[3])), str(len(b[3])), ["%03d" % wid for wid in b[3]]])
        f_w.writerow(['60-70', str(len(a[4])), str(len(b[4])), ["%03d" % wid for wid in b[4]]])
        f_w.writerow(['50-60', str(len(a[5])), str(len(b[5])), ["%03d" % wid for wid in b[5]]])
        f_w.writerow(['40-50', str(len(a[6])), str(len(b[6])), ["%03d" % wid for wid in b[6]]])
        f_w.writerow(['30-40', str(len(a[7])), str(len(b[7])), ["%03d" % wid for wid in b[7]]])
        f_w.writerow(['20-30', str(len(a[8])), str(len(b[8])), ["%03d" % wid for wid in b[8]]])
        f_w.writerow(['10-20', str(len(a[9])), str(len(b[9])), ["%03d" % wid for wid in b[9]]])
        f_w.writerow(['0-10', str(len(a[10])), str(len(b[10])), ["%03d" % wid for wid in b[10]]])

def run_model_analyzer(config, analyze_path, model):
    gt, filenames = load_data(analyze_path)
    save_file = os.path.join(config.save_dir, "analyzer_" + config.decoder_type + ".csv")
    generate_accuracy_bins(filenames, gt, model, save_file)
