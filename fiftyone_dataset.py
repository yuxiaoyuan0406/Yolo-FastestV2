import fiftyone as fo

def make_dataset():
    return fo.zoo.load_zoo_dataset(
        name='open-images-v6',
        split='train',
        label_types=['detections'],
        classes=['Bicycle', 'Motorcycle'],
        max_samples=50000
    )


if __name__ == '__main__':
    dataset = make_dataset()
    # '''
    session = fo.launch_app(dataset=dataset, port=5151)

    while True:
        try:
            pass
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt exception, closing...")
            session.close()
            # dataset.delete()
            break
    # '''

    # export_dir = '/media/dennis/DENNIS/fiftyone/open-images-v6/yolov4'
    # dataset_type = fo.types.YOLOv4Dataset

    # dataset.export(export_dir, dataset_type)
    dataset.delete()
