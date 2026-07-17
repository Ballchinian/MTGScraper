#one shared connection pool for the whole app. every request borrows a
#connection, runs its queries and hands it right back. register_vector
#teaches each connection about the pgvector column type so embeddings come
#back as arrays we can pass straight into the next query

import os

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

def setup(conn):
    register_vector(conn)
    conn.commit()  #the type lookup opens a transaction, close it or the pool complains


#railway hands you this connection string, see the readme for setup.
#min_size matches max so all four connections stay warm: a multi-line
#search borrows up to three at once for its side-by-side line scans, and
#opening a connection mid-request costs a tls handshake the visitor waits on
pool = ConnectionPool(
    os.environ["DATABASE_URL"],
    min_size=4,
    max_size=4,
    kwargs={"row_factory": dict_row},
    configure=setup,
    open=True,
)
