import psycopg2
from db_config import DB_CONFIG

def get_user_data(phone_number):
    """
    Fetches user data from the database based on the phone number.

    Args:
        phone_number (str): The user's phone number.

    Returns:
        str: The user's data formatted as required.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Ensure the temp_da_data table exists
        create_temp_table_query = """
        CREATE TEMPORARY TABLE IF NOT EXISTS temp_da_data (
            name TEXT,
            mobile BIGINT,
            MS_Assesment_started_date DATE,
            MS_Assesment_generated_date DATE,
            TimeForQ1 BIGINT, TimeForQ2 BIGINT, TimeForQ3 BIGINT, TimeForQ4 BIGINT, TimeForQ5 BIGINT,
            TimeForQ6 BIGINT, TimeForQ7 BIGINT, TimeForQ8 BIGINT, TimeForQ9 BIGINT, TimeForQ10 BIGINT,
            TimeForQ11 BIGINT, TimeForQ12 BIGINT, TimeForQ13 BIGINT, TimeForQ14 BIGINT, TimeForQ15 BIGINT,
            TimeForQ16 BIGINT, TimeForQ17 BIGINT, TimeForQ18 BIGINT, TimeForQ19 BIGINT, TimeForQ20 BIGINT,
            TimeForQ21 BIGINT, TimeForQ22 BIGINT, TimeForQ23 BIGINT, TimeForQ24 BIGINT, TimeForQ25 BIGINT,
            minMaxResult TEXT, yesCount BIGINT, noCount BIGINT, checks BIGINT, authenticity BIGINT,
            attentiveness BIGINT, selects TEXT
        );
        """
        cur.execute(create_temp_table_query)
        
        # Populate the temp_da_data table with the relevant data
        populate_temp_table_query = """
        INSERT INTO temp_da_data
        SELECT name, mobile,
               CAST(CAST("MS Assesment Started Date/Time" AS TEXT) AS DATE),
               CAST(CAST("MoneySign Generated Date" AS TEXT) AS DATE),
               CAST(CAST("Time for Q1" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q2" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q3" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q4" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q5" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q6" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q7" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q8" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q9" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q10" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q11" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q12" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q13" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q14" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q15" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q16" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q17" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q18" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q19" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q20" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q21" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q22" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q23" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q24" AS DECIMAL) AS BIGINT),
               CAST(CAST("Time for Q25" AS DECIMAL) AS BIGINT),
               NULL AS minMaxResult, 0 AS yesCount, 0 AS noCount, 0 AS checks, 0 AS authenticity,
               0 AS attentiveness, NULL AS selects
        FROM fun_getusermsdata_V2()
        WHERE "Time for Q1" IS NOT NULL AND "Time for Q2" IS NOT NULL AND "Time for Q3" IS NOT NULL
          AND "Time for Q4" IS NOT NULL AND "Time for Q5" IS NOT NULL AND "Time for Q6" IS NOT NULL
          AND "Time for Q7" IS NOT NULL AND "Time for Q8" IS NOT NULL AND "Time for Q9" IS NOT NULL
          AND "Time for Q10" IS NOT NULL AND "Time for Q11" IS NOT NULL AND "Time for Q12" IS NOT NULL
          AND "Time for Q13" IS NOT NULL AND "Time for Q14" IS NOT NULL AND "Time for Q15" IS NOT NULL
          AND "Time for Q16" IS NOT NULL AND "Time for Q17" IS NOT NULL AND "Time for Q18" IS NOT NULL
          AND "Time for Q19" IS NOT NULL AND "Time for Q20" IS NOT NULL AND "Time for Q21" IS NOT NULL
          AND "Time for Q22" IS NOT NULL AND "Time for Q23" IS NOT NULL AND "Time for Q24" IS NOT NULL
          AND "Time for Q25" IS NOT NULL;
        """
        cur.execute(populate_temp_table_query)
        
        # Query to fetch user data based on phone number
        fetch_user_data_query = """
        WITH user_answers AS (
            SELECT
                CONCAT(p.first_name, ' ', p.last_name) AS UserName,
                p.mobile_number AS MobileNumber,
                tdd.MS_Assesment_generated_date AS MS_AssesmentGeneratedDate,
                MAX(CASE WHEN q.sequence = 1 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q1,
                MAX(CASE WHEN q.sequence = 2 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q2,
                MAX(CASE WHEN q.sequence = 3 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q3,
                MAX(CASE WHEN q.sequence = 4 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q4,
                MAX(CASE WHEN q.sequence = 5 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q5,
                MAX(CASE WHEN q.sequence = 6 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q6,
                MAX(CASE WHEN q.sequence = 7 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q7,
                MAX(CASE WHEN q.sequence = 8 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q8,
                MAX(CASE WHEN q.sequence = 9 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q9,
                MAX(CASE WHEN q.sequence = 10 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q10,
                MAX(CASE WHEN q.sequence = 11 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q11,
                MAX(CASE WHEN q.sequence = 12 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q12,
                MAX(CASE WHEN q.sequence = 13 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q13,
                MAX(CASE WHEN q.sequence = 14 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q14,
                MAX(CASE WHEN q.sequence = 15 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q15,
                MAX(CASE WHEN q.sequence = 16 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q16,
                MAX(CASE WHEN q.sequence = 17 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q17,
                MAX(CASE WHEN q.sequence = 18 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q18,
                MAX(CASE WHEN q.sequence = 19 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q19,
                MAX(CASE WHEN q.sequence = 20 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q20,
                MAX(CASE WHEN q.sequence = 21 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q21,
                MAX(CASE WHEN q.sequence = 22 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q22,
                MAX(CASE WHEN q.sequence = 23 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q23,
                MAX(CASE WHEN q.sequence = 24 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q24,
                MAX(CASE WHEN q.sequence = 25 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q25,
                MAX(CASE WHEN q.sequence = 26 THEN ua.score || '_' || a.answer ELSE NULL END) AS Q26
            FROM
                temp_da_data tdd
            JOIN
                user_profile p ON tdd.mobile = p.mobile_number
            JOIN
                user_answer ua ON p.user_code = ua.user_code
            JOIN
                answer a ON a.id = ANY(string_to_array(ua.answersid, ',')::bigint[])
            JOIN
                question q ON a.questionid = q.id
            JOIN
                trait t ON q.traitid = t.id
            WHERE
                p.mobile_number = %s
            GROUP BY 
                p.first_name, p.last_name, p.mobile_number, tdd.MS_Assesment_generated_date
        )
        SELECT * FROM user_answers
        ORDER BY 
            UserName, MobileNumber, MS_AssesmentGeneratedDate
        """
        
        cur.execute(fetch_user_data_query, (phone_number,))
        result = cur.fetchone()
        
        if result:
            # Format the result as required
            user_data = " ".join([str(item) for item in result[3:] if item is not None])
            return user_data
        else:
            return None

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    finally:
        cur.close()
        conn.close()
