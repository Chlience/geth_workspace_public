export GCCL_PART_OPT="METIS"

# GAT Products

mkdir -p results/gat_ogbn-products/
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 1 --wait_time 50 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 2 --wait_time 80 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 3 --wait_time 80 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 4 --wait_time 80 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 5 --wait_time 90 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 6 --wait_time 90 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 7 --wait_time 90 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-products --size 8 --wait_time 100 2>&1 | tee results/gat_ogbn-products/no_scale_no_micro_8.txt

# GAT Reddit

mkdir -p results/gat_reddit/
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 1 --wait_time 30 2>&1 | tee results/gat_reddit/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 2 --wait_time 60 2>&1 | tee results/gat_reddit/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 3 --wait_time 60 2>&1 | tee results/gat_reddit/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 4 --wait_time 60 2>&1 | tee results/gat_reddit/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 5 --wait_time 60 2>&1 | tee results/gat_reddit/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 6 --wait_time 60 2>&1 | tee results/gat_reddit/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 7 --wait_time 70 2>&1 | tee results/gat_reddit/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_reddit --size 8 --wait_time 70 2>&1 | tee results/gat_reddit/no_scale_no_micro_8.txt

# GAT yelp

mkdir -p results/gat_yelp/
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 1 --wait_time 25 2>&1 | tee results/gat_yelp/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 2 --wait_time 40 2>&1 | tee results/gat_yelp/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 3 --wait_time 40 2>&1 | tee results/gat_yelp/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 4 --wait_time 40 2>&1 | tee results/gat_yelp/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 5 --wait_time 50 2>&1 | tee results/gat_yelp/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 6 --wait_time 50 2>&1 | tee results/gat_yelp/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 7 --wait_time 60 2>&1 | tee results/gat_yelp/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_yelp --size 8 --wait_time 60 2>&1 | tee results/gat_yelp/no_scale_no_micro_8.txt

# GAT arxiv

mkdir -p results/gat_ogbn-arxiv/
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 1 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 2 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 3 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 4 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_4.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 5 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 6 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_6.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 7 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gat_ogbn-arxiv --size 8 --wait_time 20 2>&1 | tee results/gat_ogbn-arxiv/no_scale_no_micro_8.txt

################################

# GCN Products

mkdir -p results/gcn_ogbn-products/
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 1 --wait_time 50 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 2 --wait_time 80 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 3 --wait_time 80 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 4 --wait_time 80 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 5 --wait_time 90 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 6 --wait_time 90 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 7 --wait_time 90 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-products --size 8 --wait_time 100 2>&1 | tee results/gcn_ogbn-products/no_scale_no_micro_8.txt

# GCN Reddit

mkdir -p results/gcn_reddit/
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 1 --wait_time 30 2>&1 | tee results/gcn_reddit/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 2 --wait_time 60 2>&1 | tee results/gcn_reddit/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 3 --wait_time 60 2>&1 | tee results/gcn_reddit/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 4 --wait_time 60 2>&1 | tee results/gcn_reddit/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 5 --wait_time 60 2>&1 | tee results/gcn_reddit/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 6 --wait_time 60 2>&1 | tee results/gcn_reddit/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 7 --wait_time 70 2>&1 | tee results/gcn_reddit/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_reddit --size 8 --wait_time 70 2>&1 | tee results/gcn_reddit/no_scale_no_micro_8.txt

# GCN yelp

mkdir -p results/gcn_yelp/
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 1 --wait_time 25 2>&1 | tee results/gcn_yelp/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 2 --wait_time 40 2>&1 | tee results/gcn_yelp/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 3 --wait_time 40 2>&1 | tee results/gcn_yelp/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 4 --wait_time 40 2>&1 | tee results/gcn_yelp/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 5 --wait_time 50 2>&1 | tee results/gcn_yelp/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 6 --wait_time 50 2>&1 | tee results/gcn_yelp/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 7 --wait_time 60 2>&1 | tee results/gcn_yelp/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_yelp --size 8 --wait_time 60 2>&1 | tee results/gcn_yelp/no_scale_no_micro_8.txt

# GCN arxiv

mkdir -p results/gcn_ogbn-arxiv/
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 1 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 2 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 3 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 4 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_4.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 5 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 6 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_6.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 7 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gcn_ogbn-arxiv --size 8 --wait_time 20 2>&1 | tee results/gcn_ogbn-arxiv/no_scale_no_micro_8.txt

################################

# GIN Products

mkdir -p results/gin_ogbn-products/
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 1 --wait_time 50 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 2 --wait_time 80 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 3 --wait_time 80 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 4 --wait_time 80 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 5 --wait_time 90 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 6 --wait_time 90 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 7 --wait_time 90 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-products --size 8 --wait_time 100 2>&1 | tee results/gin_ogbn-products/no_scale_no_micro_8.txt

# GIN Reddit

mkdir -p results/gin_reddit/
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 1 --wait_time 30 2>&1 | tee results/gin_reddit/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 2 --wait_time 60 2>&1 | tee results/gin_reddit/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 3 --wait_time 60 2>&1 | tee results/gin_reddit/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 4 --wait_time 60 2>&1 | tee results/gin_reddit/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 5 --wait_time 60 2>&1 | tee results/gin_reddit/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 6 --wait_time 60 2>&1 | tee results/gin_reddit/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 7 --wait_time 70 2>&1 | tee results/gin_reddit/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_reddit --size 8 --wait_time 70 2>&1 | tee results/gin_reddit/no_scale_no_micro_8.txt

# GIN yelp

mkdir -p results/gin_yelp/
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 1 --wait_time 25 2>&1 | tee results/gin_yelp/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 2 --wait_time 40 2>&1 | tee results/gin_yelp/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 3 --wait_time 40 2>&1 | tee results/gin_yelp/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 4 --wait_time 40 2>&1 | tee results/gin_yelp/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 5 --wait_time 50 2>&1 | tee results/gin_yelp/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 6 --wait_time 50 2>&1 | tee results/gin_yelp/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 7 --wait_time 60 2>&1 | tee results/gin_yelp/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_yelp --size 8 --wait_time 60 2>&1 | tee results/gin_yelp/no_scale_no_micro_8.txt

# GIN arxiv

mkdir -p results/gin_ogbn-arxiv/
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 1 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 2 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 3 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 4 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_4.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 5 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 6 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_6.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 7 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py gin_ogbn-arxiv --size 8 --wait_time 20 2>&1 | tee results/gin_ogbn-arxiv/no_scale_no_micro_8.txt

################################

# Sage Products

mkdir -p results/sage_ogbn-products/
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 1 --wait_time 50 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 2 --wait_time 80 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 3 --wait_time 80 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 4 --wait_time 80 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 5 --wait_time 90 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 6 --wait_time 90 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 7 --wait_time 90 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-products --size 8 --wait_time 100 2>&1 | tee results/sage_ogbn-products/no_scale_no_micro_8.txt

# Sage Reddit

mkdir -p results/sage_reddit/
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 1 --wait_time 30 2>&1 | tee results/sage_reddit/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 2 --wait_time 60 2>&1 | tee results/sage_reddit/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 3 --wait_time 60 2>&1 | tee results/sage_reddit/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 4 --wait_time 60 2>&1 | tee results/sage_reddit/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 5 --wait_time 60 2>&1 | tee results/sage_reddit/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 6 --wait_time 60 2>&1 | tee results/sage_reddit/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 7 --wait_time 70 2>&1 | tee results/sage_reddit/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_reddit --size 8 --wait_time 70 2>&1 | tee results/sage_reddit/no_scale_no_micro_8.txt

# Sage yelp

mkdir -p results/sage_yelp/
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 1 --wait_time 25 2>&1 | tee results/sage_yelp/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 2 --wait_time 40 2>&1 | tee results/sage_yelp/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 3 --wait_time 40 2>&1 | tee results/sage_yelp/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 4 --wait_time 40 2>&1 | tee results/sage_yelp/no_scale_no_micro_4.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 5 --wait_time 50 2>&1 | tee results/sage_yelp/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 6 --wait_time 50 2>&1 | tee results/sage_yelp/no_scale_no_micro_6.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 7 --wait_time 60 2>&1 | tee results/sage_yelp/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_yelp --size 8 --wait_time 60 2>&1 | tee results/sage_yelp/no_scale_no_micro_8.txt

# Sage arxiv

mkdir -p results/sage_ogbn-arxiv/
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 1 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_1.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 2 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_2.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 3 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_3.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 4 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_4.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 5 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_5.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 6 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_6.txt
GCCL_PART_OPT="RECUR_MINI" python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 7 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_7.txt
python3 -u tests/st/training_no_scale_cmdline.py sage_ogbn-arxiv --size 8 --wait_time 20 2>&1 | tee results/sage_ogbn-arxiv/no_scale_no_micro_8.txt
