import pandas as pd 
import numpy as np
import random as rand

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, ntile, when, lit, pandas_udf
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

from rdkit import Chem
from rdkit.Chem import Descriptors

class Downsampler:

    def __init__(self, data, sample_size=3000000, seed=42):

        self.data = data
        self.sampled_data = None
        self.sample_size = sample_size
        self.balanced_data = None
        self.seed = seed


    def compute_sample_sizes(self):

        binds_1_size = int(self.sample_size * 0.1)
        binds_0_size = self.sample_size - binds_1_size

        binds_1_sampled = (
            self.data.filter(col("binds") == 1)
            .orderBy(rand())
            .limit(binds_1_size)
        )
        binds_0_sampled = (
            self.data.filter(col("binds") == 0)
            .orderBy(rand())
            .limit(binds_0_size)
        )

        self.sampled_data = binds_0_sampled.union(binds_1_sampled)
   
    def calculate_descriptors(self):
        

        desc_schema = StructType([StructField("mol_wt", DoubleType(), True)])

        @pandas_udf(desc_schema)
        def calculate_desc(smiles_series: pd.Series) -> pd.DataFrame:
            mols = smiles_series.apply(Chem.MolFromSmiles)
            mw = mols.apply(lambda mol: round(Descriptors.MolWt(mol), 3) if mol else None)
            return pd.DataFrame({'mol_wt': mw})

        self.sampled_data = self.sampled_data.withColumn("block1", calculate_desc(col("buildingblock1_smiles")))
        self.sampled_data = self.sampled_data.withColumn("block3", calculate_desc(col("buildingblock3_smiles")))

    def transform_features(self):

        selected_columns = [
            "id",
            col("block1.mol_wt").alias("block1_mol_wt"),
            col("block3.mol_wt").alias("block3_mol_wt"),
            "binds"
        ]

        flattened_data = self.sampled_data.select(*selected_columns)
        desc_columns = [col_name for col_name in flattened_data.columns if col_name != "id" and col_name != "binds"]

        assembler = VectorAssembler(inputCols=desc_columns, outputCol="features")
        self.sampled_data = assembler.transform(flattened_data).select("id", "binds", "features")

    def bnn(self):

        num_bins = 30

        windowSpec = Window.partitionBy("desc_bin").orderBy("features")
        self.sampled_data = self.sampled_data.withColumn(
            "desc_bin", ntile(num_bins).over(windowSpec)
        )

        bin_counts = self.sampled_data.groupBy("desc_bin", "binds").count()

        bin_counts_pivot = bin_counts.groupBy("desc_bin").pivot("binds").sum("count").fillna(0)
        bin_counts_pivot = bin_counts_pivot.withColumnRenamed("0", "count_binds_0").withColumnRenamed("1", "count_binds_1")

        # Compute fraction to downsample binds=0 to match a 3:1 ratio
        bin_counts_pivot = bin_counts_pivot.withColumn(
            "fraction_binds_0",
            (col("count_binds_1") * 3) / col("count_binds_0")  )
        
        fractions = bin_counts_pivot.select("desc_bin", "fraction_binds_0").rdd.collectAsMap()

        binds_0_sampled = self.sampled_data.filter(col("binds") == 0).sampleBy("desc_bin", fractions, seed=42)

        # Keep 
        # all binds=1 (since it’s the minority class)
        binds_1 = self.sampled_data.filter(col("binds") == 1)

        # Merge sampled binds=0 with all binds=1
        self.balanced_data = binds_1.unionByName(binds_0_sampled.select(binds_1.columns))
        self.balanced_data.write.save(f"./intermediates/balanced_data_{self.sample_size}", format="parquet", mode='overwrite')

    def finalize_data(self):
        
        data_subset = self.data.select("id", "molecule_smiles")
        self.final_data = self.balanced_data.drop("features").join(data_subset, on="id", how="left")
        self.final_data.write.save(f"./intermediates/final_data_{self.sample_size}", format="parquet", mode='overwrite')

    def run(self):

        self.compute_sample_sizes()
        self.calculate_descriptors()
        self.transform_features()
        self.bnn()
        self.finalize_data()
