import psycopg2
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
        
        # get grasp batch
        grasp_batch = self.grasp_cur.fetchmany(self.batch_size)

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

 
 
if __name__ == '__main__':
    mr = ModelReader()

    batch = mr.getGraspBatch()
    for grasp_data in batch:
        print(mr.getModelInfo(grasp_data['scaled_model_id']))
        

