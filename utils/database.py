import psycopg2
import pickle
import ast
import os
from utils.config import config
 
class ModelReader:
    """ModelReader class
    
    Connects to the PSQL database for CGDB-10.0
    to fetch grasp info.

    Parameters
    ----------
    verbose : boolean, optional
        Defaults to False

    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.conn = None
        self.connect()
        self.grasp_cur = self.query()
        
    def connect(self):
        """Connect to the PSQL database"""
        try:
            # read connection parameters
            params = config()
 
            # connect to the PostgreSQL server
            if self.verbose:
                print('Connecting to the PostgreSQL database...')
            self.conn = psycopg2.connect(**params)
 
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def query(self):
        """Create the cursor"""
        grasp_cur = self.conn.cursor()
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
        grasp_cur.execute(sql_hand_grasps)
        return grasp_cur

    def prepare_sample(self, sample):
        """Puts the sample in the appropriate dict"""
        return { 
                'scaled_model_id':sample[0],
                'grasp_pregrasp_joints':sample[1], 
                'grasp_grasp_joints':sample[2], 
                'grasp_pregrasp_position':sample[3], 
                'grasp_grasp_position':sample[4], 
                'grasp_contacts':sample[5], 
                'grasp_epsilon_quality':sample[6],
                'grasp_volume_quality':sample[7]
        }


    def getGraspBatch(self, batch_size=32):
        """Gets a list of hand grasp data of size batch_size
        
        Parameters
        ----------
        batch_size : int, optional
            Defaults to 32

        Returns
        -------
        list
            Size of batch_size

        Raises
        ------
        ConnectionError
            if not connected to the PSQL database

        """
        if self.conn == None:
            raise ConnectionError("Can't get model batch, no connection")

        # Create cursor if not already exists
        if self.grasp_cur == None:
            self.grasp_cur = self.query()
        try:
            grasp_batch = list(next(self.grasp_cur) for _ in range(batch_size))
        except StopIteration:
            print ("Ran through all batches. Re-querying")
            self.grasp_cur = self.query()

        grasps = [self.prepare_sample(grasp) for grasp in grasp_batch]
        return grasps

    def getModelInfo(self, scaled_model_id):
        """
        Gets the scale of the model (for a grasp) and the path to the
        model's object file
        
        Parameters
        ----------
        scaled_model_id : int
            ID of a scaled model from a grasp

        Returns
        -------
        tuple pair of (model_scale, model_grasping_rescale, model_file_path)
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
            Select original_model_geometry_path,
            original_model_grasping_rescale
            from public.original_model
            WHERE original_model_id=""" + str(scale_dict['original_model_id'])
        cur.execute(sql_orig_model)
        res = cur.fetchone()
        model_file_path = res[0]
        model_rescale = res[1]
       
        # close the communication with the PostgreSQL
        cur.close()
        return scale_dict['scaled_model_scale'], model_rescale, model_file_path

    def getAll(self):
        """Get all information in the database"""
        grasp_cur = self.query()
        return grasp_cur.fetchall()

    def __len__(self):
        """Number of samples in the database"""
        if not hasattr(self, 'size'):
            self.size = len(self.getAll())
        return self.size


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces 
 
def show_model_points(verts, col='r', mark='o',doit=False):
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
    
    ax.scatter(x_vals, y_vals, z_vals, c=col, marker=mark)
    pyplot.show()

def getXYZ(verts,scale=1):
    print("Num points: ",len(verts))
    x_vals = list()
    y_vals = list()
    z_vals = list()
    
    for p in verts:                      
        x_vals.append(p[0]*scale)
        y_vals.append(p[1]*scale)                  
        z_vals.append(p[2]*scale)
    return x_vals, y_vals, z_vals


def show_two_plots(v1, v2):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.figure()
    ax = Axes3D(fig)
    
    ax.scatter(v1[0], v1[1], v1[2], c='r', marker='o')
    ax.scatter(v2[0], v2[1], v2[2], c='b', marker='x')
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
    mr = ModelReader()
    batch = mr.getGraspBatch(batch_size)

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

    scale,rescale, model_path = mr.getModelInfo(one_grasp["scaled_model_id"])
    model_file = open(params['model_dir'] + model_path)
    #from pyntcloud import PyntCloud
    #m = PyntCloud.from_file(params['model_dir'] + model_path)
    #print(m)
    #m = m.get_sample("points_random", n = 5000)
    #m.plot(mesh=True, backend="matplotlib")
    print(len(one_grasp['grasp_contacts']))
    contacts = one_grasp['grasp_contacts']

    verts, faces = read_off(model_file)
    #show_model_points(verts)

    print("scale: " + str(scale))
    print("rescale: " + str(rescale))

    show_two_plots(getXYZ(verts,scale*rescale), getXYZ([contacts[x:x+3] for x in range(0,len(contacts),3)]))

   
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

