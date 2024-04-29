from lib.data.datareader_fit3d import DataReaderFit3D
from lib.utils.vismo import render_and_save

def validate():
    datareader = DataReaderFit3D(n_frames=243, sample_stride=1, data_stride_test=243, data_stride_train=81)
    j2d, j3d = datareader.get_sliced_data()

    print(j2d.shape)
    print(j3d.shape)

    first_clip_2d = j2d[10]
    first_clip_3d = j3d[10]
    
    print(first_clip_2d)
    print(first_clip_3d)
    #render_and_save(first_clip_2d, "data/2d_clip.mp4")
    #render_and_save(first_clip_3d, "data/3d_clip.mp4")

if __name__ == "__main__":
    validate()