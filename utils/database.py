import psycopg2
from config import config
 
def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()
 
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
 
        # create a cursor 16713 14899
        cur = conn.cursor()
       
        batch_size = 10

        #
        # Query for grasps
        #
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

        cur.execute(sql_hand_grasps)
        
        grasp = cur.fetchone()
        # cur.fetchmany(batch_size)

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
        print(grasp_dict)

        #
        # Query scaled model
        #
        sql_scaled_model = """
            Select original_model_id, scaled_model_scale
            from public.scaled_model 
            WHERE scaled_model_id=""" + str(grasp_dict['scaled_model_id'])

        cur.execute(sql_scaled_model)
        scale_model_info = cur.fetchone()
        scale_dict = {
            'original_model_id':scale_model_info[0],
            'scaled_model_scale':scale_model_info[1],
        }
        print(scale_dict)

        #
        # Query original model
        #
        sql_orig_model = """
            Select original_model_geometry_path 
            from public.original_model
            WHERE original_model_id=""" + str(scale_dict['original_model_id'])
        cur.execute(sql_orig_model)
        model_file_path = cur.fetchone()[0]
        print(model_file_path)
       
        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
 
 
if __name__ == '__main__':
    connect()
