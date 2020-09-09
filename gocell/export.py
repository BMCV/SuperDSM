import gocell.render
import gocell.batch
import gocell.aux
import gocell.io
import gzip, dill, pathlib


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rootpath', help='root directory for batch processing')
    parser.add_argument('taskdir', help=f'batch task directory path')
    parser.add_argument('--outdir', help='output directory', default='export')
    parser.add_argument('--imageid', help='only export image ID', default=[], action='append')
    parser.add_argument('--segborder', help='border width of the segmentation masks', type=int, default=8)
    parser.add_argument('--enhance', help='apply contrast enhancement', action='store_true')
    args = parser.parse_args()
    
    rootpath = pathlib.Path(args.rootpath)
    if not rootpath.exists():
        raise ValueError(f'Root path does not exist: {rootpath}')

    taskdir = pathlib.Path(args.taskdir)
    if not taskdir.is_absolute():
        taskdir = rootpath / taskdir
    if not taskdir.is_dir():
        raise ValueError(f'Task directory does not exist: {taskdir}')

    outdir = pathlib.Path(args.outdir)
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
    data = task.run(one_shot=True, force=True, evaluation='none', out=out)

    out.write('\nRunning export:')
    for image_id in task.file_ids:
        dataframe  = data[image_id]
        outputfile = outdir / f'{image_id}.png'
        out.intermediate(f'  Processing image... {outputfile}')
        outputfile.parents[0].mkdir(parents=True, exist_ok=True)
        img = gocell.render.render_model_shapes_over_image(dataframe, border=args.segborder, normalize_img=args.enhance)
        gocell.io.imwrite(str(outputfile), img)
        out.write(f'  Exported {outputfile}')

    out.write(f'\nExported {len(task.file_ids)} files')

