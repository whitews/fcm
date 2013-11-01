"""Test reading of Coulter FCS files."""

from fcm.io import FCSreader

if __name__ == '__main__':
    bd_file = 'sample_data/3FITC_4PE_004.fcs'
    bd = FCSreader(bd_file)
    bd_data = bd.get_FCMdata()

    coulter_file = 'sample_data/coulter.fcs'
    coulter = FCSreader(coulter_file)
    coulter_data = coulter.get_FCMdata()