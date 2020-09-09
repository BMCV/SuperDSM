import gocell.render
import gocell.batch
import gocell.aux
import gocell.io
import gzip, dill, pathlib


DEFAULT_OUTDIR = {
    'gt' : 'export-gt',
    'seg': 'export-seg',
    'img': 'export-img',
    'adj': 'export-adj'
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rootpath', help='root directory for batch processing')
    parser.add_argument('taskdir', help=f'batch task directory path')
    parser.add_argument('--outdir', help='output directory', default=None)
    parser.add_argument('--imageid', help='only export image ID', default=[], action='append')
    parser.add_argument('--border', help='border width', type=int, default=8)
    parser.add_argument('--enhance', help='apply contrast enhancement', action='store_true')
    parser.add_argument('--mode', help='export the ground truth (gt), the segmentation results (seg), the raw images (img), or the adjacency graphs (adj)', default='seg')
    parser.add_argument('--gt-shuffle', help='shuffle colors of ground truth', default=[], action='append')
    args = parser.parse_args()

    if args.mode not in ('gt', 'seg', 'img', 'adj'):
        parser.error(f'Unknown mode: "{args.mode}"')
    
    rootpath = pathlib.Path(args.rootpath)
    if not rootpath.exists():
        raise ValueError(f'Root path does not exist: {rootpath}')

    taskdir = pathlib.Path(args.taskdir)
    if not taskdir.is_absolute():
        taskdir = rootpath / taskdir
    if not taskdir.is_dir():
        raise ValueError(f'Task directory does not exist: {taskdir}')

    outdir = pathlib.Path(args.outdir if args.outdir is not None else DEFAULT_OUTDIR[args.mode])
    if not outdir.is_absolute():
        outdir = taskdir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    _taskdirs = [taskdir]
    while _taskdirs[-1] != rootpath:
        _taskdirs.append(_taskdirs[-1].parents[0])

    tasks = []
    for _taskdir in _taskdirs[::-1]:
        task = gocell.batch.Task.create_from_directory(_taskdir, tasks[-1] if len(tasks) > 0 else None)
        if task is not None:
            tasks.append(task)

    out  = gocell.aux.get_output(None)
    task = tasks[-1]
    if len(args.imageid) > 0:
        task.file_ids = [gocell.batch._resolve_timings_key(file_id, task.file_ids) for file_id in args.imageid]
    task.seg_pathpattern = None
    task.log_pathpattern = None
    task.adj_pathpattern = None

    if args.mode == 'gt':
        shuffles = {}
        for shuffle_spec in args.gt_shuffle:
            image_id, shuffle_seed = shuffle_spec.split(':')
            image_id = gocell.batch._resolve_timings_key(image_id, task.file_ids)
            shuffles[image_id] = shuffle_seed
        for image_id in task.file_ids:
            outputfile = outdir / f'{image_id}.png'
            out.intermediate(f'Processing image... {outputfile}')
            gt = gocell.batch.load_gt(task.gt_loader, filepath=task.gt_pathpattern % image_id, **task.gt_loader_kwargs)
            gt = gocell.render.colorize_labels(gt, shuffle=shuffles.get(image_id, 0))
            outputfile.parents[0].mkdir(parents=True, exist_ok=True)
            gocell.io.imwrite(str(outputfile), gt)
    elif args.mode == 'img':
        for image_id in task.file_ids:
            im_filepath = str(task.im_pathpattern) % image_id
            outputfile = outdir / f'{image_id}.png'
            out.intermediate(f'Processing image... {outputfile}')
            img = gocell.io.imread(im_filepath)
            if args.enhance: img = gocell.render.normalize_image(img)
            outputfile.parents[0].mkdir(parents=True, exist_ok=True)
            gocell.io.imwrite(str(outputfile), img)
    elif args.mode in ('seg', 'adj'):
        if args.mode == 'adj': task.last_stage = 'atoms'
        data = task.run(one_shot=True, force=True, evaluation='none', out=out)
        out.write('\nRunning export:')
        for image_id in task.file_ids:
            dataframe  = data[image_id]
            outputfile = outdir / f'{image_id}.png'
            out.intermediate(f'  Processing image... {outputfile}')
            outputfile.parents[0].mkdir(parents=True, exist_ok=True)
            if args.mode == 'seg':
                img = gocell.render.render_model_shapes_over_image(dataframe, border=args.border, normalize_img=args.enhance)
            elif args.mode == 'adj':
                ymap = gocell.render.render_ymap(dataframe)[:,:,:3]
                ymap = gocell.render.render_atoms(dataframe, override_img=ymap, border_color=(0,0,0), border_radius=args.border // 2)
                img  = gocell.render.render_adjacencies(dataframe, override_img=ymap, edge_color=(0,1,0), endpoint_color=(0,1,0))
            gocell.io.imwrite(str(outputfile), img)
            out.write(f'  Exported {outputfile}')
        out.write('\n')
    out.write(f'Exported {len(task.file_ids)} files')

