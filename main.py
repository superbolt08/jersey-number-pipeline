import argparse
import os
import shutil
import subprocess
import time
import legibility_classifier as lc
import numpy as np
import json
import helpers
from tqdm import tqdm
import configuration as config
from pathlib import Path


class _StepTimer:
    """Print wall-clock delta since last line and total elapsed (for pipeline logging)."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0

    def tick(self, msg):
        now = time.perf_counter()
        dt = now - self._last
        total = now - self._t0
        self._last = now
        print(f"{msg}  [+{dt:.2f}s | {total:.1f}s total]", flush=True)


def _run_shell_with_updates(label, command, timer=None):
    """Run a shell command with unbuffered Python in the child and visible progress hints."""
    if timer:
        timer.tick(f"[{label}] subprocess starting (output follows)")
    else:
        print(f"\n[{label}] starting — subprocess output below (CPU steps can take tens of minutes).", flush=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    proc = subprocess.run(command, shell=True, env=env)
    ok = proc.returncode == 0
    if timer:
        timer.tick(f"[{label}] subprocess {'OK' if ok else 'FAILED'} (exit {proc.returncode})")
    else:
        print(f"[{label}] {'OK' if ok else 'FAILED'} (exit code {proc.returncode})", flush=True)
    return ok


def _tracklet_dir_names(images_root):
    """Names of subdirectories only (ignore .DS_Store and other files in images/)."""
    return [
        name
        for name in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, name))
    ]


def _resume_skip(force, resume, outputs_ok, step_label, timer=None):
    """If resume mode and outputs look done, skip this step."""
    if force or not resume:
        return False
    if outputs_ok:
        if timer:
            timer.tick(f"Skipping {step_label} (--resume: expected outputs already present)")
        else:
            print(f"Skipping {step_label} (--resume: expected outputs already present)", flush=True)
        return True
    return False


def get_soccer_net_raw_legibility_results(args, use_filtered = True, filter = 'gauss', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = _tracklet_dir_names(path_to_images)
    results_dict = {x:[] for x in tracklets}

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)


    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        #images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, config.dataset['SoccerNet']['legibility_model'], threshold=-1, arch=config.dataset['SoccerNet']['legibility_model_arch'])
        results_dict[directory] = track_results

    # save results
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['raw_legible_result'])
    with open(full_legibile_path, "w") as outfile:
        json.dump(results_dict, outfile)

    return results_dict

def get_soccer_net_legibility_results(args, use_filtered = False, filter = 'sim', exclude_balls=True, tracklet_ids=None):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = _tracklet_dir_names(path_to_images)

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    if tracklet_ids is not None:
        kept = set(tracklets)
        tracklets = [t for t in tracklet_ids if t in kept]

    legibility_scores = {}
    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results, track_raw = lc.run(
            images_full_path,
            config.dataset['SoccerNet']['legibility_model'],
            arch=config.dataset['SoccerNet']['legibility_model_arch'],
            threshold=0.5,
            return_raw_scores=True,
        )
        for p, s in zip(images_full_path, track_raw):
            legibility_scores[p] = float(s)
            legibility_scores[os.path.basename(p)] = float(s)
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    scores_name = config.dataset['SoccerNet'][args.part].get('legibility_scores', 'legibility_scores.json')
    legibility_scores_path = os.path.join(config.dataset['SoccerNet']['working_dir'], scores_name)
    with open(legibility_scores_path, 'w') as out_scores:
        json.dump(legibility_scores, out_scores)

    # save results
    json_object = json.dumps(legible_tracklets, indent=4)
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['legible_result'])
    with open(full_legibile_path, "w") as outfile:
        outfile.write(json_object)

    full_illegibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['illegible_result'])
    json_object = json.dumps({'illegible': illegible_tracklets}, indent=4)
    with open(full_illegibile_path, "w") as outfile:
        outfile.write(json_object)

    return legible_tracklets, illegible_tracklets


def generate_json_for_pose_estimator(args, legible = None):
    all_files = []
    if not legible is None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['SoccerNet']['root_dir'])
        image_dir = config.dataset['SoccerNet'][args.part]['images']
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = _tracklet_dir_names(path_to_images)
        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = os.listdir(track_dir)
            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['pose_input_json'])
    helpers.generate_json(all_files, output_json)


def consolidated_results(image_dir, dict, illegible_path, soccer_ball_list=None):
    if not soccer_ball_list is None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        balls_list = balls_json['ball_tracks']
        for entry in balls_list:
            dict[str(entry)] = 1

    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict['illegible']
    for entry in all_illegible:
        if not str(entry) in dict.keys():
            dict[str(entry)] = -1

    all_tracks = _tracklet_dir_names(image_dir)
    for t in all_tracks:
        if not t in dict.keys():
            dict[t] = -1
        else:
            dict[t] = int(dict[t])
    return dict

def train_parseq(args):
    if args.dataset == 'Hockey':
        print("Train PARSeq for Hockey")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
        if shutil.which("conda"):
            command = (
                f"conda run --no-capture-output -n {config.str_env} python train.py "
                f"+experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 "
                f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
            )
        else:
            command = (
                f"python train.py "
                f"+experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 "
                f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
            )
        success = _run_shell_with_updates('PARSeq training (Hockey)', command)
        os.chdir(current_dir)
        print("Done training")
    else:
        print("Train PARSeq for Soccer")
        parseq_dir = config.str_home
        current_dir = os.getcwd()
        os.chdir(parseq_dir)
        data_root = os.path.join(current_dir, config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet']['numbers_data'])
        if shutil.which("conda"):
            command = (
                f"conda run --no-capture-output -n {config.str_env} python train.py "
                f"+experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 "
                f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
            )
        else:
            command = (
                f"python train.py "
                f"+experiment=parseq dataset=real data.root_dir={data_root} trainer.max_epochs=25 "
                f"pretrained=parseq trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=2"
            )
        success = _run_shell_with_updates('PARSeq training (SoccerNet)', command)
        os.chdir(current_dir)
        print("Done training")


def hockey_pipeline(args):
    # actions = {"legible": True,
    #            "pose": False,
    #            "crops": False,
    #            "str": True}
    success = True
    # test legibility classification
    if args.pipeline['legible']:
        root_dir = os.path.join(config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["legibility_data"])

        print("Test legibility classifier")
        command = f"python3 legibility_classifier.py --data {root_dir} --arch resnet34 --trained_model {config.dataset['Hockey']['legibility_model']}"
        success = _run_shell_with_updates('Hockey legibility classifier', command)
        print("Done legibility classifier")

    if success and args.pipeline['str']:
        print("Predict numbers")
        current_dir = os.getcwd()
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
        if shutil.which("conda"):
            command = (
                f"conda run --no-capture-output -n {config.str_env} python str.py {config.dataset['Hockey']['str_model']} "
                f"--data_root={data_root}"
            )
        else:
            command = (
                f"python str.py {config.dataset['Hockey']['str_model']} "
                f"--data_root={data_root}"
            )
        success = _run_shell_with_updates('Hockey STR (PARSeq)', command)
        print("Done predict numbers")

def soccer_net_pipeline(args):
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    analysis_results = None
    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True
    timer = _StepTimer()
    timer.tick(f"SoccerNet pipeline start (part={args.part})")

    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['images'])
    soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
    features_dir = config.dataset['SoccerNet'][args.part]['feature_output_folder']
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                  config.dataset['SoccerNet'][args.part]['illegible_result'])
    gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['gt'])

    input_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                              config.dataset['SoccerNet'][args.part]['pose_input_json'])
    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_output_json'])

    part_cfg = config.dataset['SoccerNet'][args.part]
    if 'final_result' not in part_cfg or not part_cfg['final_result']:
        final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'], 'final_results.json')
    else:
        final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'], part_cfg['final_result'])

    str_result_file = os.path.join(
        config.dataset['SoccerNet']['working_dir'],
        config.dataset['SoccerNet'][args.part]['jersey_id_result'],
    )
    crops_destination_dir = os.path.join(
        config.dataset['SoccerNet']['working_dir'],
        config.dataset['SoccerNet'][args.part]['crops_folder'],
        'imgs',
    )

    resume = getattr(args, 'resume', False)
    force = getattr(args, 'force', False)

    def _all_reid_features_exist():
        if not os.path.isdir(features_dir):
            return False
        if tracklet_subset is not None:
            tracks = tracklet_subset
        else:
            tracks = [
                d for d in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, d))
            ]
        if not tracks:
            return False
        return all(
            os.path.isfile(os.path.join(features_dir, f'{t}_features.npy'))
            for t in tracks
        )

    def _gaussian_outputs_exist():
        # Defaults must match gaussian_outliers.py (threshold=3.5, rounds=3 -> r+1 in 1..3)
        return os.path.isfile(
            os.path.join(features_dir, 'main_subject_gauss_th=3.5_r=3.json')
        )

    def _pose_output_ok():
        if not os.path.isfile(output_json):
            return False
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return len(data.get('pose_results', [])) > 0
        except (json.JSONDecodeError, OSError):
            return False

    def _crops_exist():
        return os.path.isdir(crops_destination_dir) and len(os.listdir(crops_destination_dir)) > 0

    tracklet_subset = None
    subset_file = ''
    mtl = getattr(config, 'soccer_net_max_tracklets', None)
    if mtl is not None:
        all_dirs = sorted(
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        )
        tracklet_subset = all_dirs[: int(mtl)]
        subset_file = os.path.join(
            config.dataset['SoccerNet']['working_dir'],
            f'tracklet_subset_{args.part}.json',
        )
        with open(subset_file, 'w') as sf:
            json.dump(tracklet_subset, sf)
        timer.tick(
            f"SoccerNet tracklet limit: {len(tracklet_subset)} of {len(all_dirs)} folders (see {subset_file})"
        )

    subset_arg = ''
    if tracklet_subset is not None:
        subset_arg = f' --subset_file "{os.path.abspath(subset_file)}"'

    # 1. Filter out soccer ball based on images size
    if args.pipeline['soccer_ball_filter']:
        if not _resume_skip(force, resume, os.path.isfile(soccer_ball_list), 'soccer ball detection', timer):
            timer.tick("Determine soccer ball")
            success = helpers.identify_soccer_balls(
                image_dir, soccer_ball_list, allowed_tracklets=tracklet_subset
            )
            timer.tick("Done determine soccer ball")

    # 1. generate and store features for each image in each tracklet
    if args.pipeline['feat']:
        if not _resume_skip(force, resume, _all_reid_features_exist(), 'ReID feature extraction', timer):
            timer.tick("Generate features")
            if shutil.which("conda"):
                command = (
                    f"conda run --no-capture-output -n {config.reid_env} python {config.reid_script} "
                    f"--tracklets_folder {image_dir} --output_folder {features_dir}{subset_arg}"
                )
            else:
                command = (
                    f"python {config.reid_script} "
                    f"--tracklets_folder {image_dir} --output_folder {features_dir}{subset_arg}"
                )
            success = _run_shell_with_updates('ReID feature extraction', command, timer=timer)
            timer.tick("Done generating features")

    #2. identify and remove outliers based on features
    if args.pipeline['filter'] and success:
        if not _resume_skip(force, resume, _gaussian_outputs_exist(), 'Gaussian outlier filtering', timer):
            timer.tick("Identify and remove outliers")
            command = f"python gaussian_outliers.py --tracklets_folder {image_dir} --output_folder {features_dir}{subset_arg}"
            success = _run_shell_with_updates('Gaussian outlier filtering', command, timer=timer)
            timer.tick("Done removing outliers")

    #3. pass all images through legibililty classifier and record results
    if args.pipeline['legible'] and success:
        legible_ready = os.path.isfile(full_legibile_path) and os.path.isfile(illegible_path)
        if _resume_skip(force, resume, legible_ready, 'legibility classification', timer):
            pass
        else:
            timer.tick("Classifying legibility")
            try:
                legible_dict, illegible_tracklets = get_soccer_net_legibility_results(
                    args, use_filtered=True, filter='gauss', exclude_balls=True, tracklet_ids=tracklet_subset
                )
                #get_soccer_net_raw_legibility_results(args)
                #legible_dict, illegible_tracklets = get_soccer_net_combined_legibility_results(args)
            except Exception as error:
                print(f'Failed to run legibility classifier:{error}')
                success = False
            timer.tick("Done classifying legibility")

    #3.5 evaluate tracklet legibility results
    if args.pipeline['legible_eval'] and success:
        timer.tick("Evaluate legibility results")
        try:
            if legible_dict is None:
                 with open(full_legibile_path, 'r') as openfile:
                    # Reading from json file
                    legible_dict = json.load(openfile)

            helpers.evaluate_legibility(gt_path, illegible_path, legible_dict, soccer_ball_list=soccer_ball_list)
        except Exception as e:
            print(e)
            success = False
        timer.tick("Done evaluating legibility")


    #4. generate json for pose-estimation
    if args.pipeline['pose'] and success:
        if _resume_skip(force, resume, _pose_output_ok(), 'pose (JSON + ViTPose inference)', timer):
            pass
        else:
            timer.tick("Generating json for pose")
            try:
                if legible_dict is None:
                    with open(full_legibile_path, 'r') as openfile:
                        # Reading from json file
                        legible_dict = json.load(openfile)
                generate_json_for_pose_estimator(args, legible = legible_dict)
            except Exception as e:
                print(e)
                success = False
            timer.tick("Done generating json for pose")

            # 4.5 Alternatively generate json for pose for all images in test/train
            #generate_json_for_pose_estimator(args)


            #5. run pose estimation and store results
            if success:
                timer.tick("Detecting pose (ViTPose subprocess)")
                if shutil.which("conda"):
                    command = (
                        f"conda run --no-capture-output -n {config.pose_env} python pose.py "
                        f"{config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py "
                        f"{config.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} "
                        f"--out-json {output_json}"
                    )
                else:
                    command = (
                        f"python pose.py "
                        f"{config.pose_home}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py "
                        f"{config.pose_home}/checkpoints/vitpose-h.pth --img-root / --json-file {input_json} "
                        f"--out-json {output_json}"
                    )
                success = _run_shell_with_updates('ViTPose (pose.py)', command, timer=timer)
                timer.tick("Done detecting pose")


    #6. generate cropped images
    if args.pipeline['crops'] and success:
        if not _resume_skip(force, resume, _crops_exist(), 'torso crop generation', timer):
            timer.tick("Generate crops")
            try:
                Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
                if legible_results is None:
                    with open(full_legibile_path, "r") as outfile:
                        legible_results = json.load(outfile)
                scores_name = config.dataset['SoccerNet'][args.part].get('legibility_scores', 'legibility_scores.json')
                legibility_scores_path = os.path.join(config.dataset['SoccerNet']['working_dir'], scores_name)
                legibility_scores = None
                if os.path.isfile(legibility_scores_path):
                    with open(legibility_scores_path, 'r') as sf:
                        legibility_scores = json.load(sf)
                prop = config.proposal
                helpers.generate_crops(
                    output_json,
                    crops_destination_dir,
                    legible_results,
                    legibility_scores=legibility_scores,
                    min_legibility_score=prop['min_legibility_score_for_crop'],
                    use_color_filter=prop['use_color_filter_on_crops'],
                    crop_width_scale=prop.get('crop_width_scale', 1.0),
                )
            except Exception as e:
                print(e)
                success = False
            timer.tick("Done generating crops")

    #7. run STR system on all crops
    if args.pipeline['str'] and success:
        if not _resume_skip(force, resume, os.path.isfile(str_result_file), 'STR (PARSeq inference)', timer):
            timer.tick("Predict numbers (STR)")
            crops_data_root = os.path.join(
                config.dataset['SoccerNet']['working_dir'],
                config.dataset['SoccerNet'][args.part]['crops_folder'],
            )
            min_str = config.proposal['min_str_frame_confidence']
            letterbox_arg = ' --letterbox_pad' if config.proposal.get('str_letterbox_pad', False) else ''
            if shutil.which("conda"):
                command = (
                    f"conda run --no-capture-output -n {config.str_env} python str.py {config.dataset['SoccerNet']['str_model']} "
                    f"--data_root={crops_data_root} --batch_size=1 --inference --result_file {str_result_file} "
                    f"--min_str_confidence={min_str}{letterbox_arg}"
                )
            else:
                command = (
                    f"python str.py {config.dataset['SoccerNet']['str_model']} "
                    f"--data_root={crops_data_root} --batch_size=1 --inference --result_file {str_result_file} "
                    f"--min_str_confidence={min_str}{letterbox_arg}"
                )
            success = _run_shell_with_updates('STR / PARSeq inference', command, timer=timer)
            if success:
                timer.tick("Done predict numbers (STR)")
            else:
                timer.tick(
                    f"STR failed — in conda env '{config.str_env}' run: "
                    f"pip install lmdb  (and str/parseq: pip install -r requirements/core.txt; pip install -e .). "
                    f"Or set configuration.str_env to an env with torch + strhub + lmdb."
                )

    #str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'], "val_jersey_id_predictions.json")
    if args.pipeline['combine'] and success:
        if _resume_skip(force, resume, os.path.isfile(final_results_path), 'tracklet combine / final_results.json', timer):
            pass
        else:
            #8. combine tracklet results (proposal: digit-wise logits or confidence-weighted aggregation)
            prop = config.proposal
            mfc = prop['min_tracklet_frame_confidence']
            if prop['combine_mode'] == 'digit_wise':
                results_dict, analysis_results = helpers.process_jersey_id_predictions_bayesian(
                    str_result_file, useTS=False, useBias=True, useTh=False
                )
            else:
                results_dict, analysis_results = helpers.process_jersey_id_predictions(
                    str_result_file, useBias=True, min_frame_confidence=mfc
                )

            # add illegible tracklet predictions
            consolidated_dict = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list=soccer_ball_list)

            with open(final_results_path, 'w') as f:
                json.dump(consolidated_dict, f)
            timer.tick("Wrote final_results / consolidated tracklet predictions")

    if args.pipeline['eval'] and success:
        #9. evaluate accuracy
        timer.tick("Evaluate tracklet accuracy vs ground truth")
        if consolidated_dict is None:
            with open(final_results_path, 'r') as f:
                consolidated_dict = json.load(f)
        with open(gt_path, 'r') as gf:
            gt_dict = json.load(gf)
        print(len(consolidated_dict.keys()), len(gt_dict.keys()))
        helpers.evaluate_results(consolidated_dict, gt_dict, full_results = analysis_results)
        timer.tick("SoccerNet pipeline finished")
    elif success:
        timer.tick("SoccerNet pipeline finished (eval step disabled)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Options: 'SoccerNet', 'Hockey'")
    parser.add_argument('part', help="Options: 'test', 'val', 'train', 'challenge")
    parser.add_argument('--train_str', action='store_true', default=False, help="Run training of jersey number recognition")
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help="SoccerNet only: skip a stage if its output files already exist (saves time on re-runs).",
    )
    parser.add_argument(
        '--force',
        action='store_true',
        default=False,
        help="SoccerNet only: re-run every stage even if outputs exist (overrides --resume).",
    )
    args = parser.parse_args()

    if not args.train_str:
        if args.dataset == 'SoccerNet':
            actions = {"soccer_ball_filter": True,
                       "feat": True,
                       "filter": True,
                       "legible": True,
                       "legible_eval": False,
                       "pose": True,
                       "crops": True,
                       "str": True,
                       "combine": True,
                       "eval": True}
            args.pipeline = actions
            soccer_net_pipeline(args)
        elif args.dataset == 'Hockey':
            actions = {"legible": True,
                       "str": True}
            args.pipeline = actions
            hockey_pipeline(args)
        else:
            print("Unknown dataset")
    else:
        train_parseq(args)


