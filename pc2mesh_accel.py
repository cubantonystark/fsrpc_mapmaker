# Fast Surface Reconstruction for PointClouds using NKSR
# To be included with Mapmaker
# Version 4.0. Copyright 2024, Reynel Rodriguez and EolianVR, Inc.

import torch, os, zipfile, utm, glob, shutil, pymeshlab, sys, logging, pymeshlab
import open3d as o3d
import numpy as np
from pycg import vis, exp
from nksr import Reconstructor, utils, fields
from PIL import Image
from pyntcloud import PyntCloud

import subprocess

level = logging.INFO
format = '%(message)s'
handlers = [logging.StreamHandler()]
logging.basicConfig(level=level, format='%(asctime)s \033[1;34;40m%(levelname)-8s \033[1;37;40m%(message)s',
                    datefmt='%H:%M:%S', handlers=handlers)

class pc2mesh_gpu():
    
    def kill_process(self):
        
        ftrs = [f for f in glob.glob(os.getcwd()+"/pointclouds/*")]
        
        for ftr in ftrs:
            
            os.remove(ftr)
            
        output = subprocess.check_output('ps -A | grep bash', shell=True).decode()
        
        output = output.split('\n')
        
        try:
        
            for pids in output:
                
                output = pids.split()
        
                os.system("kill -9 "+str(output[0])) 
                
        except IndexError:
            
            pass
            
    def create_artak_files(self, o_filename, d_filename, lat, lon):
        
        face_number = 100000
        texture_size = 8192
        
        #print("Generating ARTAK bundle")
        logging.info("Generating ARTAK bundle")
        
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(d_filename)
        m = ms.current_mesh()
        
        # This portion does the decimation to a set amount of faces
        
        f_number = m.face_number()
        
        c = 1
        
        while f_number > face_number:
            
            m = ms.current_mesh()
            f_number = m.face_number()
            target_faces = int(f_number / 1.5)
            
            #print("Target: " + str(int(target_faces)) + " F. Iter. " + str(c) + ".")
            logging.info("Target: " + str(int(target_faces)) + " F. Iter. " + str(c) + ".")
            
            ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                            targetfacenum=int(target_faces), targetperc=0,
                            qualitythr=0.3,
                            optimalplacement=True,
                            preservenormal=True,
                            autoclean=True)    
            
            f_number = m.face_number()
            v_number = m.vertex_number()
            ratio = (abs(target_faces / f_number) - 1.1) * 10  # Efficiency ratio. resulting amt faces vs target amt of faces   
            
            #print('Achieved: ' + str(f_number) + ' F. Ratio ==> ' + '%.2f' % abs(ratio) + ':1.00.')
            logging.info('Achieved: ' + str(f_number) + ' F. Ratio ==> ' + '%.2f' % abs(ratio) + ':1.00.')
            c += 1
    
        m = ms.current_mesh()
        f_number = m.face_number()
        v_number = m.vertex_number()
        
        #print('End VC: ' + str(v_number) + '. End FC: ' + str(f_number) + ".") 
        logging.info('End VC: ' + str(v_number) + '. End FC: ' + str(f_number) + ".")
        
        ms.save_current_mesh(d_filename,
                             save_vertex_color=True,
                             save_vertex_coord=True,
                             save_vertex_normal=True,
                             save_face_color=False,
                             save_wedge_texcoord=False,
                             save_wedge_normal=False,
                             save_polygonal=False)    
        
        # This portion will generate the textures for the mesh using the texture baking process.
        
        ms = pymeshlab.MeshSet()
        
        ms.load_new_mesh(o_filename)
        ms.load_new_mesh(d_filename)    
        
        #print("Parametrization with " + str(texture_size))
        logging.info("Parametrization with " + str(texture_size))
        
        ms.apply_filter('compute_texcoord_parametrization_triangle_trivial_per_wedge',
                        sidedim=0,
                        textdim=texture_size,
                        border=3,
                        method='Basic')
        
        percentage = pymeshlab.PercentageValue(2)
        
        ms.apply_filter('transfer_attributes_to_texture_per_vertex',
                        sourcemesh=0,
                        targetmesh=1,
                        attributeenum=0,
                        upperbound=percentage,
                        textname='texture.png',
                        textw=texture_size,
                        texth=texture_size,
                        overwrite=False,
                        pullpush=True)
        
        ms.save_current_mesh(o_filename,
                             save_vertex_color=False,
                             save_vertex_coord=False,
                             save_vertex_normal=False,
                             save_face_color=True,
                             save_wedge_texcoord=True,
                             save_wedge_normal=True,
                             save_polygonal=False) 
        
        os.remove(d_filename)
        
        # We need to compress the texture file
        img = Image.open('pointclouds/texture.png')
        img = img.convert("P", palette=Image.WEB, colors=256)
        img.save('pointclouds/texture.png', optimize=True)
            
        extensions = ['.obj', '.obj.mtl', '.png', '.xyz', '.prj']
        
        compression = zipfile.ZIP_DEFLATED
        zip_file = o_filename.replace('.obj', '')+ '.zip'    
        
        with zipfile.ZipFile(zip_file, mode="w") as zf:
            
            for ext in extensions:
                try:
                    zf.write(o_filename.replace('.obj', '')+ext, o_filename.replace('.obj', '').replace('pointclouds/', '')+ ext, compress_type=compression, compresslevel=9)
                    
                except FileExistsError:
                    pass
                except FileNotFoundError:
                    pass
                
            zf.write('pointclouds/texture.png', 'texture.png', compress_type=compression, compresslevel=9)

    def main(self):
        
        if os.path.exists('pointlouds'):
            shutil.rmtree('pointclouds')
            os.mkdir('pointclouds')
    
        def load_e57(filename):
    

    
            return
        
        filename = [f for f in glob.glob(os.getcwd()+"/pointclouds/*")]
        
        try:
            
            filename = filename[0]
            print(filename)
            
        except IndexError:
        
            #print("No file passed")
            logging.info("No file passed")
            
            self.kill_process()
        
        ply_point_cloud = o3d.data.PLYPointCloud()
        
        # Convert to PLY in case this is a .pts file or an e57 one
        
        if filename.endswith(".pts"):
            
            logging.info("Converting to PLY")
        
            pcd = o3d.io.read_point_cloud(filename)
        
            pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
        
            o3d.io.write_point_cloud(filename.replace("pts", "ply"), pcd)
        
            filename = filename.replace("pts", "ply")        
            
        elif filename.endswith(".e57"):

            logging.info("Converting to PLY")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)

            ms.save_current_mesh(filename.replace('e57', 'ply'),
                                 save_vertex_color=True,
                                 save_vertex_coord=True,
                                 save_vertex_normal=True,
                                 save_face_color=True,
                                 save_wedge_texcoord=True,
                                 save_wedge_normal=True)

            filename = filename.replace("e57", "ply")  
            
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"]= "1"
        
        torch.cuda.empty_cache() 
 
        ply_point_cloud = o3d.data.PLYPointCloud()
            
        path, file = os.path.split(str(filename))
        
        #print("Loading PointCloud")
        #logging.info("Loading PointCloud")        
            
        pcd = o3d.io.read_point_cloud(str(filename))
            
        #print("Downsampling")
        logging.info("Downsampling")
            
        downpcd = pcd.voxel_down_sample(voxel_size = 0.1)
            
        filename = path+"/"+"ds_"+file
            
        o3d.io.write_point_cloud(filename, downpcd)
            
        # We will attempt to use NKSR and build a mesh from the PointCloud
            
        try:
            
            test_geom = vis.from_file(filename)
            
            device = torch.device("cuda:0")
            
            input_xyz = torch.from_numpy(np.asarray(test_geom.points)).float().to(device)
            input_normal = torch.from_numpy(np.asarray(test_geom.normals)).float().to(device)
            input_color = torch.from_numpy(np.asarray(test_geom.colors)).float().to(device)
            
            #print("GPU Reconstruction")
            logging.info("GPU Reconstruction")
            
            nksr = Reconstructor(device)
            
            Reconstructor.chunk_tmp_device = torch.device("cpu")
            
            # Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
            #field = nksr.reconstruct(input_xyz, input_normal, chunk_size = 50.0, detail_level = 1.0)
            
            field = nksr.reconstruct(input_xyz, input_normal, chunk_size = 25.0, approx_kernel_grad = True, solver_tol = 1e-4, fused_mode = True, detail_level = 1.0)
            
            field.set_texture_field(fields.PCNNField(input_xyz, input_color))
            
            mesh = field.extract_dual_mesh()
            mesh = vis.mesh(mesh.v, mesh.f, color = mesh.c)
            
            #ftr = filename
            
            path, file = os.path.split(filename)
            
            #vis.show_3d([mesh])
            
            mesh.triangle_normals = o3d.utility.Vector3dVector([])
            
            file = "pointclouds/"+file.replace("ply", "obj")
            input_xyz = torch.from_numpy(np.asarray(test_geom.points)).float().to(device)
            input_normal = torch.from_numpy(np.asarray(test_geom.normals)).float().to(device)
            
            file = file.replace("ds_", "")
            
            o3d.io.write_triangle_mesh(file, mesh, write_triangle_uvs = True, print_progress = True)
            
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(file) 
            
            #print("Refining")
            logging.info("Refining")
            
            p = pymeshlab.PercentageValue(25)
            
            # The selection process and removal of long faces will create floaters, we will remove isolated faces
            ms.apply_filter('meshing_remove_connected_component_by_diameter', mincomponentdiag = p)        
            
            filename = filename.replace('.ply', '.obj').replace('ds_', '')
            
            ms.save_current_mesh(file,
                                 save_vertex_color=True,
                                 save_vertex_coord=True,
                                 save_vertex_normal=True,
                                 save_face_color=True,
                                 save_wedge_texcoord=True,
                                 save_wedge_normal=True,
                                 save_polygonal=True)        
            
            o_filename = file
            d_filename = o_filename.replace('pointclouds/', 'pointclouds/decimated_')
            
            tgt_folder = o_filename.replace('.obj', '').replace('pointclouds/', '')
            
            shutil.copy(o_filename, d_filename)
            
        except (RuntimeError, pymeshlab.pmeshlab.PyMeshLabException) as err:         
            
            # Should the reconstruction fail using NKSR, we will use Open3D on the CPU to reconstruct that way
            
            #print("\nThis Reconstruction exceeds available GPU Memory. Switiching to CPU Reconstruction.\n")
            #logging.info("\nThis PointCloud is not suitable for GPU Reconstruction. Switching to CPU.")
            
            with open('/mnt/d/Projects/map_maker_1_2/ARTAK_MM/LOGS/gpu_status.txt', 'w') as msg:
                
                msg.write('gpu_fail')
            
            self.kill_process()
            
            sys.exit()

        # We will encode the lat and lon into utm compliant coordinates for the xyz file and retrieve the utm zone for the prj file
        
        try:
    
            utm_easting, utm_northing, zone, zone_letter = utm.from_latlon(float(lat), float(lon))
            utm_easting = "%.2f" % utm_easting
            utm_northing = "%.2f" % utm_northing
            
        except (utm.error.OutOfRangeError, UnboundLocalError) as err:
            
            #print("Coordinate values out of range. Will encode lat=0, lon=0.")
            logging.info("Coordinate values out of range. Will encode lat=0, lon=0.")
            
            lat = 0
            lon = 0
            
        utm_easting, utm_northing, zone, zone_letter = utm.from_latlon(float(lat), float(lon))
        utm_easting = "%.2f" % utm_easting
        utm_northing = "%.2f" % utm_northing                
        
        # Create xyz and prj based on lat and lon provided
        prj_1 = 'PROJCS["WGS 84 / UTM zone '
        prj_2 = '",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-81],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32617"]]'
    
        with open(o_filename.replace('.obj','.xyz'), 'w') as xyz:
            xyz.write(str(utm_easting + " " + str(utm_northing) + " " + "101.000"))
    
        with open(o_filename.replace('.obj','.prj'), 'w') as prj:
            prj.write(str(prj_1) + str(zone) + str(prj_2))            
        
        self.create_artak_files(o_filename, d_filename, lat, lon)
        
        zip_file = o_filename.replace('.obj', '')+ '.zip'
        
        tgt_folder = o_filename.replace('.obj', '').replace('pointclouds/', '')
        
        #shutil.copy(zip_file, '/mnt/d/Projects/map_maker_1_2/ARTAK_MM/DATA/PointClouds/LowRes/'+zip_file.replace('pointclouds/', ''))
        
        shutil.copy(zip_file, '/mnt/c/map_maker_1_2/ARTAK_MM/POST/Lidar/'+tgt_folder+'/Data/'+zip_file.replace('pointclouds/', ''))
        
        os.remove(zip_file)
        
        extensions = ['.obj', '.obj.mtl', '.xyz', '.prj']
        
        for ext in extensions:
            
            shutil.copy(o_filename.replace('.obj', ext), '/mnt/c/map_maker_1_2/ARTAK_MM/POST/Lidar/'+tgt_folder+'/Data/Model')         
         
            shutil.copy('pointclouds/texture.png', '/mnt/c/map_maker_1_2/ARTAK_MM/POST/Lidar/'+tgt_folder+'/Data/Model/texture.png')
        
        with open('/mnt/c/Projects/map_maker_1_2/ARTAK_MM/LOGS/gpu_status.txt', 'w') as msg:
            
            msg.write('gpu_success')
                
        logging.info("GPU Reconstruction complete.")
        
        self.kill_process()
        
if __name__ == '__main__':
    
    pc2mesh_gpu().main()
