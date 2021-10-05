import fiftyone as fo

if __name__ == '__main__':
    dataset = fo.zoo.load_zoo_dataset('open-images-v6', 
        split='train', 
        label_types=['detections'], 
        classes=['Bicycle', 'Motorcycle'],
        max_samples=500)
    session = fo.launch_app(dataset=dataset, port=5151)

    while True:
        try:
            pass
        except KeyboardInterrupt as e:
            print(e)
            session.close()
            break
