from volume_calculator import *
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def read_off(filename, material=2, unit='cm'):
    myOFFUtils = OFFUtils()
    myOFFUtils.loadOFF(filename)
    material_mass = materialsFor3DPrinting().get_material_mass(material)
    v = myOFFUtils.volume(unit, material_mass)
    m = myOFFUtils.mass(unit, material_mass)
    a = myOFFUtils.area()
    print(v,m,a)

def read_stl(filename, material=2, unit='cm'):
    mySTLUtils = STLUtils()
    mySTLUtils.loadSTL(filename)
    material_mass = materialsFor3DPrinting().get_material_mass(material)
    v = mySTLUtils.volume(unit, material_mass)
    m = mySTLUtils.mass(unit, material_mass)
    a = mySTLUtils.area()
    t = mySTLUtils.triangles
    return v,m,a,t

if __name__ == '__main__':
    DATA_DIR = '/data/csc4801/KedzioraLab'
    read_stl(DATA_DIR+'/thingiverse/vase/stls/Decorative_Vase/vase_flower.stl')

    names = ['airplane' , 'bookshelf' , 'bottle' , 'bowl' , 'car' , 'cone' ,
    'cup' , 'dresser' , 'stairs' , 'vase' , 'table' , 'flower_pot']

    partition = 'thingiverse'
    df = pd.DataFrame()
    all_tags = []
    all_desc = []
    if partition == 'thingiverse':
        for name in names:
            for folder in glob.glob(os.path.join(DATA_DIR,partition,name,'stls','*'), recursive=True):
                try:
                    with open(folder+'/TAGS.txt') as file:
                        tags = [line.rstrip() for line in file]
                    with open(folder+'/DESCRIPTIONS.txt') as file:
                        desc = ' '.join(line.rstrip() for line in file)
                    all_tags += tags
                    all_desc += [desc]
                except Exception as e:
                    pass
                stls = glob.glob(os.path.join(folder,'*.stl'))
                for stl in stls:
                    try:
                        (v,m,a,t) = read_stl(stl)
                        entry = pd.DataFrame({'class':name, 'path':stl, 'volume':v, 'mass':m, 'area':a, 'triangles':[t]}, index=[0]) 
                        df = pd.concat([df, entry], ignore_index=True)
                    except Exception as e:
                        print(e)
                        pass
    else:
        for name in names:
            stls = glob.glob(os.path.join(DATA_DIR,partition,name,'**','*.stl'), recursive=True)
            for stl in stls:
                try:
                    (v,m,a,t) = read_stl(stl)
                    entry = pd.DataFrame({'class':name, 'path':stl, 'volume':v,
                        'mass':m, 'area':a, 'triangles':[t]}, index=[0])
                    df = pd.concat([df, entry], ignore_index=True)
                except Exception as e:
                    print(e)
                    pass

    def print_fig():
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
        print(my_base64_jpgData,end='.')

    # Get the violin plot of the volume
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.violinplot(df['volume'])
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Volume ($cm^3$)")
    ax.set_title("Volume of Thingiverse STLs")
    print_fig()

    # get a violin plot of the surface areas
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.violinplot(df['area'])
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Surface Area ($cm^2$)")
    ax.set_title("Surface Area of Thingiverse STLs")
    print_fig()

    # get a violin plot of the surface areas
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.violinplot(df['mass'])
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Expected Mass (kg)")
    ax.set_title("Mass of Thingiverse STLs (using PLA)")
    print_fig()

    # get the distribution of classes
    fig = plt.figure()
    ax = df['class'].value_counts().plot(kind='barh')
    ax.set_xlabel("Count of STLs")
    ax.set_ylabel("Class of Object")
    ax.set_title("Count of STLs per Class")
    print_fig()

    # get a barplot of the tags
    fig = plt.figure()
    ax = pd.Series(all_tags).value_counts().plot(kind='barh')
    ax.set_xlabel("Thingiverse Tags")
    ax.set_ylabel("Count of Tag")
    ax.set_title("Count of Tags per Thingiverse STL")
    print_fig()

    # get the distribution of the triangle areas
    util = STLUtils()
    sa = [util.areaOfTriangle(*t) for ts in df['triangles'] for t in ts]
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.violinplot(sa)
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Triangle Mesh Surface Area ($cm^2$)")
    ax.set_title("Surface Area of Thingiverse STL Triangles")
    print_fig()



    # df.to_csv(DATA_DIR+'/'+partition+'_stats.csv')
