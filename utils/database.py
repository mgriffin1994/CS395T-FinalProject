import psycopg2
import pickle
import ast
import os
from config import config
 
class ModelReader:
    
    def __init__(self, batch_size=32):
        self.conn = None
        self.grasp_cur = None
        self.batch_size = batch_size
        self.connect()
        
    def connect(self):
        try:
            # read connection parameters
            params = config()
 
            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            self.conn = psycopg2.connect(**params)
 
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def getGraspBatch(self):
        """
        @returns A list of hand grasp data of size batch_size
        """
        if self.conn == None:
            print("Can't get model batch, no connection")

        # Create cursor if not already exists
        if self.grasp_cur == None:
            self.grasp_cur = self.conn.cursor()
            #self.grasp_cur.itersize = self.batch_size
            # Query for grasps
            sql_hand_grasps = """
                                Select scaled_model_id, 
                                grasp_pregrasp_joints, 
                                grasp_grasp_joints, 
                                grasp_pregrasp_position, 
                                grasp_grasp_position, 
                                grasp_contacts, 
                                grasp_epsilon_quality,
                                grasp_volume_quality
                                from public.grasp WHERE hand_id=4
                              """
            self.grasp_cur.execute(sql_hand_grasps)
        
        grasp_batch = list(next(self.grasp_cur) for _ in range(self.batch_size))

        grasps = list()
        for grasp in grasp_batch:
            grasp_dict = {
                'scaled_model_id':grasp[0],
                'grasp_pregrasp_joints':grasp[1], 
                'grasp_grasp_joints':grasp[2], 
                'grasp_pregrasp_position':grasp[3], 
                'grasp_grasp_position':grasp[4], 
                'grasp_contacts':grasp[5], 
                'grasp_epsilon_quality':grasp[6],
                'grasp_volume_quality':grasp[7]
            }
            grasps.append(grasp_dict) 

        return grasps

    def getModelInfo(self, scaled_model_id):
        """
        Gets the scale of the model (for a grasp) and the path to the
        model's object file
        @params scaled_model_id ID of a scaled model from a grasp
        @returns tuple pair of (model_scale, model_file_path)
        """
        cur = self.conn.cursor()

        # Query scaled model
        sql_scaled_model = """
            Select original_model_id, scaled_model_scale
            from public.scaled_model 
            WHERE scaled_model_id=""" + str(scaled_model_id)

        cur.execute(sql_scaled_model)
        scale_model_info = cur.fetchone()
        scale_dict = {
            'original_model_id':scale_model_info[0],
            'scaled_model_scale':scale_model_info[1],
        }

        # Query original model
        sql_orig_model = """
            Select original_model_geometry_path 
            from public.original_model
            WHERE original_model_id=""" + str(scale_dict['original_model_id'])
        cur.execute(sql_orig_model)
        model_file_path = cur.fetchone()[0]
       
        # close the communication with the PostgreSQL
        cur.close()
        return scale_dict['scaled_model_scale'], model_file_path


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces 
 
def show_model_points(verts):
    print("Num points: ",len(verts))
    x_vals = list()
    y_vals = list()
    z_vals = list()
    
    for p in verts:
        x_vals.append(p[0])
        y_vals.append(p[1])
        z_vals.append(p[2])

    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = pyplot.figure()
    ax = Axes3D(fig)
    
    ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')
    pyplot.show()

py_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    """
        scaled_model_id (int) 
        grasp_pregrasp_joints (float []) 
        grasp_grasp_joints (float [])
        grasp_pregrasp_position 
        grasp_grasp_position
        grasp_contacts
        grasp_epsilon_quality
        grasp_volume_quality
    """

    batch_size = 128
    mr = ModelReader(batch_size)
    batch = mr.getGraspBatch()

    #i = 0
    #while len(batch) > 0:
    #    batch = mr.getGraspBatch()
    #    i += 1
    #    print(i)


    #
    # Draw a model from a grasp
    #
    one_grasp = batch[0]
    params = config(section='data')

    scale, model_path = mr.getModelInfo(one_grasp["scaled_model_id"])
    model_file = open(params['model_dir'] + model_path)
    from pyntcloud import PyntCloud
    m = PyntCloud.from_file(params['model_dir'] + model_path)
    print(m)
    m.plot(mesh=True, backend="matplotlib")


    #verts, faces = read_off(model_file)
    #show_model_points(verts)
   
    #
    # Write joints to file
    #
    #graspfile = open("grasp_joints.txt", "w")
    #while len(batch) > 0:
    #    batch = mr.getGraspBatch()
    #    for grasp_data in batch:
    #        graspfile.write("%s\n" % str(grasp_data['grasp_grasp_joints']))
    #    i+=1
    #    print(i)
    #graspfile.close()

    #
    # Read joints from file
    #
    #graspfile = open("grasp_joints.txt", "r")
    #print(ast.literal_eval(graspfile.readline()))

