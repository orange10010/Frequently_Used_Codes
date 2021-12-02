def set_tf_device(device):
    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == 'gpu':
        print("Training on GPU...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        gpus = tf.config.list_physical_devices(device_type='GPU') # 或者tf.config.experimental.list_physical_devices("GPU"),效果应该一样。
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)

set_tf_device('gpu') # 'cpu' or 'gpu'
