include("HaloMatchingFunctions.jl")
using CSV
using Base.Threads
using DataFrames

@assert nthreads() > 1 "Julia is only using 1 thread."

# Get SLURM array task ID from environment variable
slurm_array_task_id = get(ENV, "SLURM_ARRAY_TASK_ID", nothing)
if slurm_array_task_id === nothing
    println("Warning: SLURM_ARRAY_TASK_ID is not set. Using default snap value for local testing.")
    snap = 0  # default snap value for local testing
else
    task_id = parse(Int64, slurm_array_task_id)
    println("SLURM Array Task ID: ", task_id)
    snap = task_id  # Specify the number you want
end

# Define directory paths using joinpath for consistency
directory_path_ahf = joinpath("/disk04", "wcui", "AHF-Simba-m100n1024")
directory_path_caesar = joinpath("/disk04", "kevin", "m100n1024", "AHF", "particlelists")
directory_path_output = joinpath("/disk04", "kevin", "m100n1024", "AHF", "caesarmatches")

println("Loading AHF data for snap ", snap, " ...")
file_path_ahf = get_AHF_file(directory_path_ahf, snap)
AHF_data = read_file_to_structure(file_path_ahf)
println("AHF data loaded.")

println("Loading Caesar data for snap ", snap, " ...")
file_path_caesar = get_caesar_file(directory_path_caesar, snap)
caesar_data = load_hdf5_to_namedata(file_path_caesar)
println("Caesar data loaded.")

println("Finding best matches between Caesar and AHF data ...")
best_matches = find_best_matches(caesar_data, AHF_data)
println("Matching completed.")

summarize_matches(best_matches, caesar_data, AHF_data)

# Check if all matches are -1 and print a warning if so
if all(t -> t[2] == -1, best_matches)
    println("Warning: All matches have AHFID = -1. No valid matches found.")
end

# Prepare output filename and path
full_filename = "Simba_M200_snap_" * lpad(string(snap), 3, '0') * ".csv"
filename = joinpath(directory_path_output, full_filename)

# Convert matches to DataFrame and sort by CaesarID
df = DataFrame(map(t -> (CaesarID = t[1], AHFID = t[2]), best_matches))
sort!(df, names(df)[1])

println("Writing matched data to CSV file: ", filename)
CSV.write(filename, df)
println("File writing completed.")