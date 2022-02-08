import psycopg2
def commitToDatabase(cursor,records):
    for row in records:
        cursor.execute('''Update compdata Set uber_id = %s where id = %s''',(row.comp1_id,row.id))
    return records
    
