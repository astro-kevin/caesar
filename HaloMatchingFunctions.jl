# Module for handling halo matching functions including reading particle data files,
# loading HDF5 datasets, and finding best matches between halo lists.

using ProgressMeter
using Serialization
using HDF5
using Base.Threads
using Glob
using Logging
using Statistics

"""
    ParticleMembership

Struct to hold halo ID and sets of particle IDs for different particle types.
Fields default to empty sets.
"""
Base.@kwdef struct ParticleMembership
    id::Int64
    parttype0::Set{Int64} = Set{Int64}()
    parttype1::Set{Int64} = Set{Int64}()
    parttype4::Set{Int64} = Set{Int64}()
    parttype5::Set{Int64} = Set{Int64}()
end

ParticleMembership(id) = ParticleMembership(id = id)

"""
    read_file_to_structure(file_path::AbstractString; lines::Vector{String}=Vector{String}(), max_lines=typemax(Int64))

Read a particle data file and parse it into a vector of `ParticleMembership` structs.

# Arguments
- `file_path::AbstractString`: Path to the input file.
- `lines::Vector{String}` (optional): Pre-read lines to parse instead of reading from file.
- `max_lines` (optional): Maximum number of lines to process.

# Returns
- `Vector{ParticleMembership}`: Parsed data grouped by ID with particle sets by type.

Parses lines that may contain IDs followed by particle data lines separated by tabs.
Ignores empty lines and malformed lines with debug messages.
"""
function read_file_to_structure(file_path::AbstractString; lines::Vector{String}=Vector{String}(), max_lines=typemax(Int64))
    # Read lines from the file if not provided
    if isempty(lines)
        lines = readlines(file_path)
    end

    if isempty(lines)
        println("File is empty.")
        return ParticleMembership[]  # Return an empty vector if no data
    end

    end_line = min(length(lines), max_lines)
    progress = Progress(end_line, 1, "Parsing lines")

    results = ParticleMembership[]
    current_membership = ParticleMembership(0)
    current_id = nothing
    expected_particles = 0
    particle_lines_parsed = 0
    i = 1

    while i <= end_line
        next!(progress)
        line = strip(lines[i])

        # Skip empty lines
        if isempty(line)
            i += 1
            continue
        end

        # Check if line is a group header: two integers separated by space
        split_line = split(line)
        if length(split_line) == 2
            # Try to parse group header: number_of_particles and id
            try
                expected_particles = parse(Int64, split_line[1])
                current_id = parse(Int64, split_line[2])
                current_membership = ParticleMembership(current_id)
                particle_lines_parsed = 0
                i += 1

                # Read exactly expected_particles lines of particle data
                while particle_lines_parsed < expected_particles && i <= end_line
                    particle_line = strip(lines[i])
                    # Skip empty lines within particle data (if any)
                    if isempty(particle_line)
                        i += 1
                        continue
                    end
                    # Parse particle line with tab separator
                    try
                        parts = split(particle_line, '\t', limit=2)
                        if length(parts) != 2
                            error("Invalid particle line format at line $i: $particle_line")
                        end
                        particle = parse(Int64, parts[1])
                        particletype_num = parse(Int64, parts[2])
                        if particletype_num == 0
                            push!(current_membership.parttype0, particle)
                        elseif particletype_num == 1
                            push!(current_membership.parttype1, particle)
                        elseif particletype_num == 4
                            push!(current_membership.parttype4, particle)
                        elseif particletype_num == 5
                            push!(current_membership.parttype5, particle)
                        end
                        particle_lines_parsed += 1
                    catch
                        error("Invalid particle line at line $i: $particle_line")
                    end
                    i += 1
                end

                # Assert that exactly expected_particles lines were parsed
                if particle_lines_parsed != expected_particles
                    error("Expected $expected_particles particle lines but parsed $particle_lines_parsed for group with id $current_id")
                end

                # After particle block, check next non-empty line if any
                if i <= end_line
                    next_line = strip(lines[i])
                    # Skip blank lines to find next meaningful line
                    while isempty(next_line) && i < end_line
                        i += 1
                        next_line = strip(lines[i])
                    end
                    # If not at end, check if next_line is a split marker or group header
                    if i <= end_line
                        # split marker is a line with one or more tabs and no spaces
                        if contains(next_line, '\t') && !contains(next_line, ' ')
                            # It's a split marker, ok
                        else
                            # Check if group header: two integers separated by space
                            parts_next = split(next_line)
                            if length(parts_next) == 2
                                try
                                    _ = parse(Int64, parts_next[1])
                                    _ = parse(Int64, parts_next[2])
                                    # valid group header, ok
                                catch
                                    error("Unexpected line after group at line $i: $next_line")
                                end
                            else
                                error("Unexpected line after group at line $i: $next_line")
                            end
                        end
                    end
                end

                # Save current group data
                push!(results, current_membership)
            catch e
                @debug "Skipping invalid group header line at line $i: $line"
                i += 1
            end
        else
            # Unexpected line format outside group header
            @debug "Skipping unexpected line at line $i: $line"
            i += 1
        end
    end

    return results
end


"""
    load_hdf5_to_namedata(filename::String)

Load HDF5 file datasets into an array of `ParticleMembership` structs.

# Arguments
- `filename::String`: Path to the HDF5 file.

# Returns
- `Vector{ParticleMembership}`: Array of `ParticleMembership` with particle sets loaded from datasets.

Assumes dataset names are numeric strings corresponding to `id` field.
Only loads particle type 4 data.
"""
function load_hdf5_to_namedata(filename::String)
    # Open the HDF5 file to get the number of datasets
    dataset_names = String[]
    h5open(filename, "r") do h5file
        dataset_names = keys(h5file) |> collect
    end
    
    # Preallocate the array for ParticleMembership structures
    num_datasets = length(dataset_names)
    membership_structures = Vector{ParticleMembership}(undef, num_datasets)
    
    # Open the file again for reading
    h5open(filename, "r") do h5file
        @showprogress "Processing datasets..." for (i, dataset_name) in enumerate(dataset_names)
            # Initialize ParticleMembership struct for each dataset
            data = read(h5file[dataset_name])
            particles_set = Set(Int64.(data))

            # Assuming the id corresponds to the dataset name numerically, unless otherwise specified
            id = parse(Int64, dataset_name)  # This assumes dataset names are numeric

            # Store the ParticleMembership instance into the preallocated array
            membership_structures[i] = ParticleMembership(id = id, parttype4 = particles_set)
        end
    end

    return membership_structures
end



"""
    find_best_matches(list1::Vector{ParticleMembership}, list2::Vector{ParticleMembership})

Find best matching halos between two lists of `ParticleMembership` based on particle overlap.

# Arguments
- `list1::Vector{ParticleMembership}`: First list of ParticleMembership to match from.
- `list2::Vector{ParticleMembership}`: Second list of ParticleMembership to match against.

# Returns
- `Vector{Tuple{Int64, Int64}}`: Vector of tuples `(id1, id2)` representing matches.
  If no suitable match is found, the second element is `-1`.

Matches are based on intersection size of particle sets of type `parttype4`.
A match is accepted immediately if intersection size is greater than 50% of the first set.
"""
function find_best_matches(list1::Vector{ParticleMembership}, list2::Vector{ParticleMembership})
    best_matches = Vector{Tuple{Int64, Int64}}(undef, length(list1))
    N = length(list1)
    progress = Progress(N, desc="Matching Progress")

    # Parallelized matching loop over first list
    Threads.@threads for i in 1:length(list1)
        pm1 = list1[i]
        best_intersection_size = 0
        best_match_j = 0

        set1 = pm1.parttype4

        # Iterate over second list to find best match
        for j in 1:length(list2)
            pm2 = list2[j]
            set2 = pm2.parttype4

            intersection = intersect(set1, set2)
            intersection_size = length(intersection)

            # Accept match immediately if intersection > 50% of set1
            if intersection_size > 0.5 * length(set1)
                best_match_j = j
                break
            elseif intersection_size > best_intersection_size
                best_intersection_size = intersection_size
                best_match_j = j
            end
        end

        # Store the match only if a valid index was found
        # Use -1 for no match found
        if best_match_j > 0
            best_matches[i] = (pm1.id, list2[best_match_j].id)
        else
            best_matches[i] = (pm1.id, -1)  # -1 indicates no match found
        end
        next!(progress)
        flush(stdout)
    end
    println()

    return best_matches
end

"""
    get_caesar_file(directory::String, number::Int64)

Construct the file path for a Caesar snapshot file given a directory and snapshot number.

# Arguments
- `directory::String`: Directory containing the files.
- `number::Int64`: Snapshot number.

# Returns
- `String`: Full path to the Caesar snapshot file.

Throws an error if the file does not exist.
"""
function get_caesar_file(directory::String, number::Int64)
    # Ensure the number is formatted as a 3-digit number, e.g., 001, 010, 123
    num_string = lpad(number, 3, '0')
    file_prefix = "Simba_M200_snap_"
    file_extension = ".h5"
    
    # Construct the full filename
    full_filename = file_prefix * num_string * file_extension
    file_path = joinpath(directory, full_filename)
    
    # Check if the file exists
    if isfile(file_path)
        return file_path
    else
        error("File $file_path does not exist.")
    end
end

"""
    get_AHF_file(directory::String, number::Int64)

Find the AHF particle file matching a given snapshot number in a directory.

# Arguments
- `directory::String`: Directory containing the files.
- `number::Int64`: Snapshot number.

# Returns
- `String`: Path to the matching AHF particle file.

Throws an error if no matching file is found.
"""
function get_AHF_file(directory::String, number::Int64)
    # Ensure the number is formatted as a 3-digit number
    num_string = lpad(number, 3, '0')
    file_prefix = "Simba_M200_snap_"
    file_extension = ".AHF_particles"

    # Build the desired pattern string that captures the base form
    search_pattern = file_prefix * num_string * ".z*" * file_extension
    files = glob(search_pattern, directory)

    # Filter files to ensure they match the desired float precision format
    for file in files
        if occursin(r"\.z\d+\.\d{3}\.AHF_particles$", file)
            return file  # Return the first properly qualified file
        end
    end

    error("No matching file found for number $number in the specified directory.")
end

function summarize_matches(best_matches, caesar_data, ahf_data)
    total_galaxies = length(best_matches)
    matched = filter(x -> x[2] != -1, best_matches)
    num_matched = length(matched)
    percent_matched = 100 * num_matched / total_galaxies

    println("Number of galaxies matched: $num_matched / $total_galaxies ($(round(percent_matched, digits=2))%)")

    ahf_dict = Dict(pm.id => pm for pm in ahf_data)
    caesar_dict = Dict(pm.id => pm for pm in caesar_data)

    overlap_fractions = Float64[]
    for (caesar_id, ahf_id) in matched
        caesar_set = get(caesar_dict, caesar_id, ParticleMembership(caesar_id)).parttype4
        ahf_set = get(ahf_dict, ahf_id, ParticleMembership(ahf_id)).parttype4
        if !isempty(caesar_set)
            intersection_size = length(intersect(caesar_set, ahf_set))
            overlap_fraction = intersection_size / length(caesar_set)
            push!(overlap_fractions, overlap_fraction)
        end
    end

    if !isempty(overlap_fractions)
        mean_overlap = mean(overlap_fractions)
        median_overlap = median(overlap_fractions)
        min_overlap = minimum(overlap_fractions)
        max_overlap = maximum(overlap_fractions)
        println("Overlap fraction statistics (based on parttype4):")
        println("  Mean: $(round(mean_overlap, digits=4))")
        println("  Median: $(round(median_overlap, digits=4))")
        println("  Min: $(round(min_overlap, digits=4))")
        println("  Max: $(round(max_overlap, digits=4))")
    else
        println("No overlap fractions to report.")
    end

    ahf_match_counts = Dict{Int64, Int64}()
    for (_, ahf_id) in matched
        ahf_match_counts[ahf_id] = get(ahf_match_counts, ahf_id, 0) + 1
    end

    repeated_matches = count(v -> v > 1, values(ahf_match_counts))
    println("Number of repeated halo matches: $repeated_matches")

    sorted_matches = sort(collect(ahf_match_counts), by = x -> x[2], rev=true)
    top5 = first(sorted_matches, min(5, length(sorted_matches)))
    println("Top 5 most frequently matched halos:")
    for (halo_id, count) in top5
        println("  Halo ID $halo_id matched $count times")
    end
end
"""
    read_single_halo_from_file(file_path::AbstractString, halo_id::Int64) :: Union{ParticleMembership,Nothing}

Reads a specific halo by `halo_id` from the particle data file and returns a single `ParticleMembership` object.
Returns `nothing` if the halo is not found.
Efficient for targeted queries in large files.

# Arguments
- `file_path::AbstractString`: Path to the input file.
- `halo_id::Int64`: ID of the halo to read.

# Returns
- `ParticleMembership`: The `ParticleMembership` for the halo if found, else `nothing`.
"""
function read_single_halo_from_file(file_path::AbstractString, halo_id::Int64)::Union{ParticleMembership,Nothing}
    open(file_path, "r") do io
        while !eof(io)
            line = strip(readline(io))
            if isempty(line)
                continue
            end
            split_line = split(line)
            if length(split_line) == 2
                expected_particles = try
                    parse(Int64, split_line[1])
                catch
                    continue
                end
                current_id = try
                    parse(Int64, split_line[2])
                catch
                    continue
                end
                if current_id == halo_id
                    pm = ParticleMembership(current_id)
                    particle_lines_parsed = 0
                    while particle_lines_parsed < expected_particles && !eof(io)
                        particle_line = strip(readline(io))
                        if isempty(particle_line)
                            continue
                        end
                        parts = split(particle_line, '\t', limit=2)
                        if length(parts) != 2
                            continue
                        end
                        particle = parse(Int64, parts[1])
                        particletype_num = parse(Int64, parts[2])
                        if particletype_num == 0
                            push!(pm.parttype0, particle)
                        elseif particletype_num == 1
                            push!(pm.parttype1, particle)
                        elseif particletype_num == 4
                            push!(pm.parttype4, particle)
                        elseif particletype_num == 5
                            push!(pm.parttype5, particle)
                        end
                        particle_lines_parsed += 1
                    end
                    return pm
                else
                    # skip particle lines for unwanted halo
                    for _ in 1:expected_particles
                        if !eof(io)
                            readline(io)
                        end
                    end
                end
            end
        end
    end
    return nothing
end