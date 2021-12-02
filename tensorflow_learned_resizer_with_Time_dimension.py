if method == 'method1':
        frames_final = []
        for i in range(input_.shape[1]):
            frames_tmp = input_[:,i] # (bs, H, W, C) -> (bs, 320, 320, 3)
            frames_tmp = learned_resizer(frames_tmp) # (bs, H', W', C) -> (bs, 224, 224, 3)
            frames_tmp = tf.expand_dims(frames_tmp, axis=1) # (bs, 1, H', W', C) -> (bs, 1, 224, 224, 3)
            frames_final.append(frames_tmp)

        frames_want = frames_final[0]
        for i in range(len(frames_final)):
            if i == 0:
                continue
            else:
                frames_want = tf.concat((frames_want,frames_final[i]), axis=1)
        
        frames_tmp = None
        frames_final = None

        return frames_want

    if method == 'method2':
        frames_final = []
        for i in range(input_.shape[1]):
            frames_tmp = input_[:,i] # (bs, H, W, C) -> (bs, 320, 320, 3)
            frames_tmp = learned_resizer(frames_tmp) # (bs, H', W', C) -> (bs, 224, 224, 3)
            frames_final.append(frames_tmp)

        frames_want = tf.stack(frames_final, axis=1)

        # frames_tmp = None
        # frames_final = None

        return frames_want

    if method == 'method3':
        x = input_
        batch_size, time_steps, H, W, C = x.shape
        # reshape input  to be (batch_size * timesteps, input_size)
        # x = x.contiguous().view(batch_size * time_steps, H, W, C)
        x = tf.reshape([batch_size * time_steps, H, W, C])
        x = learned_resizer(x)
        x = tf.reshape([batch_size, time_steps, 224, 224, 3])
        # x = x.contiguous().view(batch_size, time_steps, H, W, C)
    
        return x
