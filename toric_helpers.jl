using Oscar
using Polymake

using InvertedIndices
using Combinatorics
using LinearAlgebra

function slicematrix(A::Matrix{T}) where T
    m, n = size(A)
    B = Vector{T}[Vector{T}(undef, n) for _ in 1:m]
    for i in 1:m
        B[i] .= A[i, :]
    end
    return B
end

isinvertible(A) = !isapprox(Float64(det(A)), 0)

"""
    encode(::Union{Polymake.VectorAllocated{Polymake.Rational},Polymake.VectorAllocated{Polymake.Integer},Vector{Int64}})

    Internal function that converts a vector, representing a ray in the fan,
to a string in order to allow for hashing for the dictionary.

#Examples
```jldoctest
julia> encode([1,0,2,5])
"1,0,2,5"
```
"""
function encode(objects::AbstractVector)
    return(foldl((x,y) -> string(x, ',', y), objects))
end


function ration(A)
    return map(x->QQ(x), A)
end

function rational_conversion(A::Union{Array{Int64},Array{QQFieldElem}})
    L = Array{QQFieldElem}([])
    for n in A
        push!(L, QQ(n))
    end
    return L
end

function normalize(A::Array{QQFieldElem})
    if A[1] == 0
        return A
    else 
        return Array{QQFieldElem}(map(x-> QQ(x//A[1]), A))
    end
end

function fixSubObjects(L)
    len = size(L,1)
    return [getindex(L,i) for i in [1:1:len;]]
end

function fixVector(v)
    return [Int(v[i]) for i in [1:1:size(v,1);]]
end

function getRays(fan::NormalToricVariety)
    rays_pl = rays(fan) # rays but the way that polymake presents it
    
    rays_as_list = Vector{Vector{QQFieldElem}}([])
    for j in [1:1:length(rays_pl);]
        some_ray = Vector{QQFieldElem}([])
        for i in [1:1:length(getindex(rays_pl,j));]
            push!(some_ray, getindex(rays_pl,j)[i])
        end
        push!(rays_as_list, some_ray)
    end
    rays_resolved = map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),rays_as_list)
    return rays_resolved
end

function getCones(fan::NormalToricVariety)
    cones_matrix = cones(fan)
    dim = size(cones_matrix,1)
    list_of_lists = Array{Array{Int64}}([])
    for i in [1:1:dim;]
        set_indices = Set{Int64}(Polymake.row(cones_matrix,i))
        list_indices = Array{Int64}([])
        for j in set_indices
            push!(list_indices, j)
        end
        push!(list_of_lists, list_indices)
    end
    list_of_lists = map(x -> sort(x),list_of_lists)
    return list_of_lists
end

function getMaximalCones(fan::NormalToricVariety)
    cones_matrix = ray_indices(maximal_cones(fan))
    dim = size(cones_matrix,1)
    list_of_lists = Array{Array{Int64}}([])
    for i in [1:1:dim;]
        set_indices = Set{Int64}(Polymake.row(cones_matrix,i))
        list_indices = Array{Int64}([])
        for j in set_indices
            push!(list_indices, j)
        end
        push!(list_of_lists, list_indices)
    end
    list_of_lists = map(x-> sort(x), list_of_lists)
    return list_of_lists
end

"""
    Structure to store information of a stacky fan - this is a fan together with a dictionary assigning stacky values to each ray.

# Properties:
-> `fan` - the underlying fan, as a polymake object
-> `scalars` - the array of stacky values
-> `stacks` - a dictionary assigning stacky values to each ray.

"""
struct StackyFan
    fan::NormalToricVariety
    stacks::Dict{Vector{QQFieldElem}, QQFieldElem}
    # Constructors for the StackyFan object
    StackyFan(
        fan::NormalToricVariety,
        stacks::Dict{Vector{QQFieldElem}, QQFieldElem}) = new(fan, stacks)
    StackyFan(
        rays::Array{Int64, 2},
        cones::Array{Array{Int64, 1}, 1},
        scalars::Array{QQFieldElem, 1}) = makeStackyFan(rays, cones, scalars)
    StackyFan(
        fan::NormalToricVariety,
        scalars::Array{QQFieldElem, 1}) = addStackStructure(fan, scalars)
end

function rowMinors(A::AbstractMatrix{<:Union{Number, QQFieldElem}},S::Union{AbstractSet,AbstractVector})
    outList=[]
    slices=slicematrix(A)
    for i in 1:size(slices,1)
        if i in S
            append!(outList,[slices[i]])
        end
    end
    return Array(transpose(hcat(outList...)))
end
"""
    makeStackyFan(::Array{Int64,2},::Array{Array{Int64,1},1},::Array{Int64,1}))

Function to generate a stacky fan from a matrix representing rays as row vectors, a vector of vectors representing the rays contained in each cone, and a vector of stacky values to be assigned the rays. The second input should be zero-indexed.

# Examples
```jldoctest
julia> makeStackyFan([1 0; 1 1; 1 2],[[0,1],[1,2]],[2,2,2])
[ 2 ,  2 ,  2 ]
```
"""
function makeStackyFan(
    rays::Array{Int64,2},
    cones::Array{Array{Int64,1},1},
    scalars::Union{Array{Int64,1},Dict{Vector{QQFieldElem}, QQFieldElem}})

    # Construct a normal fan from the given rays and cones
    fan = normal_toric_variety(IncidenceMatrix(cones),rays, non_redundant = true)
    
    if typeof(scalars)== Dict{Vector{QQFieldElem}, QQFieldElem} # Construct the dictionary
        return StackyFan(fan, scalars)
        
    else
        rays_as_list = slicematrix(rays)
        rays_converted = []
        for L in rays_as_list
            L_conv = rational_conversion(L)
            push!(rays_converted, L_conv)
        end
        
        rays_normalized = Array{Array{QQFieldElem}}(map(x ->  normalize(x), rays_converted))
    
        scalars_conv = rational_conversion(scalars)
        pairs = map((x,y) -> (x,y), rays_normalized, scalars_conv)
        stacks = Dict(pairs)
        
        return StackyFan(fan, stacks)
    end
end

"""
    addStackStructure(::Polymake.BigObjectAllocated, ::Array{Int64, 1})

Function to generate a stacky fan from a given fan and a set of scalars.

# Examples
```jldoctest
julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 1 1; 1 2],INPUT_CONES=[[0,1],[1,2]]);

julia> stackyWeights(addStackStructure(X,[2,2,2]))
[ 2 ,  2 ,  2 ]
```
"""
function addStackStructure(
    fan::NormalToricVariety,
    scalars::Union{Array{Int64, 1},Array{QQFieldElem, 1}})

    rays_pl = getRays(fan) # rays but the way that polymake presents it
    
    rays_as_list = Vector{Vector{QQFieldElem}}([])
    for j in [1:1:length(rays_pl);]
        some_ray = Vector{QQFieldElem}([])
        for i in [1:1:length(getindex(rays_pl,j));]
            push!(some_ray, getindex(rays_pl,j)[i])
        end
        push!(rays_as_list,  some_ray)
    end
    
    scalars_in_QQ = rational_conversion(scalars)
    pairs = map((x,y) -> (x,y), rays_as_list, scalars_in_QQ)
    stacks = Dict(pairs)
    return StackyFan(fan, stacks)
end


"""
    stackyWeights(::StackyFan)

    Returns a list of the stacky weights of the rays of the given stacky fan with the same order as the rays of the fan.

#Examples
```jldoctest
julia> F=makeStackyFan([1 0; 1 1; 1 2; 1 3],[[0,1],[1,2],[2,3]],[1,2,3,4]);

julia> stackyWeights(F)
[ 1 ,  2 ,  3 ,  4 ]
```
"""
function stackyWeights(sf::StackyFan)
    dictionary = sf.stacks
    weights = Array{QQFieldElem}([])

    fan = sf.fan
    rays_pl = getRays(fan) # rays but the way that polymake presents it
    
    rays_as_list = Vector{Vector{QQFieldElem}}([])
    for j in [1:1:length(rays_pl);]
        some_ray = Vector{QQFieldElem}([])
        for i in [1:1:length(getindex(rays_pl,j));]
            push!(some_ray, getindex(rays_pl,j)[i])
        end
        push!(rays_as_list, some_ray)
    end

    out = Array{QQFieldElem}([])
    for ray in rays_as_list
        push!(out, dictionary[ray])
    end
    return out
end

"""
    getRayStack(::StackyFan, ::Array{Int64, 1})

    Get the scalar associated with a ray in the given stacky fan structure.

# Examples
```jldoctest
julia> F=makeStackyFan([1 0; 1 1; 1 2; 1 3],[[0,1],[1,2],[2,3]],[1,2,3,4]);

julia> getRayStack(F,[1,2])
3
```
"""
function getRayStack(sf::StackyFan, ray::Array{Int64, 1})
    ray_conv = Array{QQFieldElem}([])
    for i in ray
        push!(ray_conv, QQ(i))
    end
    return sf.stacks[ray_conv]
end


"""
    rootConstruction(::StackyFan, ::Array{Int64, 1})

Given a fan and a set of scalars corresponding to the rays of the fan,
performs a root construction on the fan by multiplying the stack scalars
by the given values. 

rootConstruction returns a new StackyFan object, and does not modify the input.

# Examples
```jldoctest

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0 0;1 1 0;1 0 1;1 1 1],INPUT_CONES=[[0,1,2],[1,2,3]]);

julia> SX = StackyFan(X, [2,3,5,7]);

julia> stackyWeights(rootConstruction(SX, [1, 4, 2, 1]))
[ 2 ,  12 ,  10 ,  7 ]
```
"""
function rootConstruction(
    sf::StackyFan,
    scalars::Array{QQFieldElem, 1})

    scalars_conv = Array{QQFieldElem}([])
    for i in scalars
            push!(scalars_conv, QQ(i))
    end
    # Multiply the scalars of the fan by the given values
    return addStackStructure(sf.fan, map(x -> x, stackyWeights(sf) .* scalars_conv))
end


"""
    rootConstructionDistinguishedIndices(::StackyFan, ::Array{Int64, 1}, ::Array{Int64, 1})

    Given a fan, the indices of the distinguished rays in the fan rays (as an incidence matrix), and
a set of scalars corresponding to the rays of the fan, performs a root 
construction on the fan by multiplying the stack scalars by the given values. 

    rootConstructionDistinguishedIndices returns a new StackyFan object,
and does not modify the input.

# Examples
```jldoctest
julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0 0;1 1 0;1 0 1;1 1 1],INPUT_CONES=[[0,1,2],[1,2,3]]);

julia> SX = StackyFan(X, [2,3,5,7]);

julia> stackyWeights(rootConstructionDistinguishedIndices(SX, [0,1,1,0], [4, 2, 1, 3]))
[ 2 ,  6 ,  5 ,  7 ]
```
"""
function rootConstructionDistinguishedIndices(
    sf::StackyFan,
    distIndices::Array{Int64, 1},
    scalars::Union{Array{Int64, 1},Array{QQFieldElem, 1}})
    
    numRays = size(getRays(sf.fan), 1)
    fullScalars = fill(QQ(1), numRays)
    for i in 1:numRays
        if distIndices[i]==1 && scalars[i] != 0
            fullScalars[i] = scalars[i]
        end
    end
    # Multiply the scalars of the fan by the given values
    return rootConstruction(sf, fullScalars)
end



"""
    rootConstructionDistinguished(
        ::StackyFan, 
        ::Polymake.Matrix{Polymake.Rational},
        ::Array{Int64, 1})

    Given a fan, a set of distinguished rays, and a set of scalars of equal size,
performs a root construction on the fan on the distinguished rays by multiplying 
the stack scalars by the given values.

    rootConstructionDistinguished returns a new StackyFan object, 
and does not modify the input.

# Examples
```jldoctest
julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0 0;1 1 0;1 0 1;1 1 1],INPUT_CONES=[[0,1,2],[1,2,3]]);

julia> SX = StackyFan(X, [2,3,5,7]);

julia> distinguished = X.RAYS[[2,3],:];

julia> stackyWeights(rootConstructionDistinguished(SX, distinguished, [4, 2]))
[ 2 ,  12 ,  10 ,  7 ]
```
"""
function rootConstructionDistinguished(
    sf::StackyFan,
    rays::Array{QQFieldElem,2},
    scalars::Array{QQFieldElem, 1})


    
    encoded_rays = slicematrix(rays)
    # Make a copy of the dictionary
    newStacks = copy(sf.stacks)
    for i in 1:length(encoded_rays)
        ray = encoded_rays[i]
        # Multiply the scalar of the corresponding ray
        newStacks[ray] *= scalars[i]
    end
    
    # Convert the dictionary to an array of scalars matching the indices
    #newScalars = mapslices(ray -> newStacks[encode(ray)], sf.fan.RAYS, dims=2)
    newScalars = Array{QQFieldElem, 1}()
    for i in 1:size(getRays(sf.fan), 1)
        push!(newScalars, newStacks[getRays(sf.fan)[i,:][1]])
    end
    
    return StackyFan(sf.fan, newScalars)
end



"""
    findBarycenter(::Union{AbstractSet,AbstractVector},::Polymake.BigObjectAllocated)

    Takes a normal toric variety X and a set s corresponding to a subset of rays of X, and outputs a polymake vector corresponding to the barycenter of those rays.

# Examples
```jldoctest makeSmoothWithDependencies
julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0;1 1; 0 1],INPUT_CONES=[[0,1],[1,2]]);

julia> s=[1,2];

julia> findBarycenter(s,X)
pm::Matrix<pm::Integer>
2 1
```
"""
function findBarycenter(s::Union{AbstractSet,AbstractVector},X::NormalToricVariety)
    all_rays = getRays(X)
    rayss = [all_rays[i] for i in s]
    dim = size(rayss[1], 1)
    bary = [0 for i in [1:1:dim;]]
    for i in [1:1:size(rayss,1);]
        bary = [bary[j] + rayss[i][j] for j in [1:1:dim;]]
    end
    return bary
end

"""
    findStackyBarycenter(::Union{AbstractSet,AbstractVector},::StackyFan)

    Takes a stacky fan SX and a set s corresponding to a subset of rays of SX, calculates the 'stacky rays' corresponding to those rays (the rays times their stacky values), and find the barycenter of the stacky rays.

# Examples
```jldoctest
julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 1 2],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[2,3]);

julia> findStackyBarycenter([1,2],F)
[ 5 ,  6 ]
```
"""

function findStackyBarycenter(s::Union{AbstractSet,AbstractVector},SX::StackyFan)
    X = SX.fan
    scalars_dict = SX.stacks
    all_rays =getRays(X)
    rays_subset = [Array{QQFieldElem}(Polymake.common.primitive(all_rays[i])) for i in s]
    scalars = [scalars_dict[i] for i in all_rays]
    scalars_subset = [scalars[i] for i in s]

    rayss = [scalars_subset[i] *rays_subset[i] for i in [1:1:size(rays_subset,1);]]
    dim = size(rays_subset[1], 1)
    bary = [0 for i in [1:1:dim;]]
    for i in [1:1:size(rays_subset,1);]
        bary = [bary[j] + rayss[i][j] for j in [1:1:dim;]]
    end
    return bary
end


function rowMinors(A::AbstractMatrix{<:Union{Number, QQFieldElem}},S::Union{AbstractSet,AbstractVector})
    outList=[]
    slices=slicematrix(A)
    for i in 1:size(slices,1)
        if i in S
            append!(outList,[slices[i]])
        end
    end
    return Array(transpose(hcat(outList...)))
end

function myTranspose(A::Vector{Vector{QQFieldElem}})
    B= Vector{Vector{QQFieldElem}}([])
    colSize = size(A[1],1)
    rowSize = size(A, 1)
    for i in [1:1:colSize;]
        col = Vector{QQFieldElem}([])
        for j in [1:1:rowSize;]
            push!(col,A[j][i])
        end
        push!(B, col)
    end
    return B
end

function newRowMinors(A::Vector{Vector{QQFieldElem}}, S::Union{AbstractSet,AbstractVector})
    outList = Array{Array{QQFieldElem}}([])
    for i in S
        push!(outList,A[i])
    end
    return outList
end

function getConeRank(coneRayIndices::AbstractVector, rayMatrix::Vector{Vector{QQFieldElem}})
    return rank(matrix(QQ,reduce(hcat,newRowMinors(rayMatrix, coneRayIndices))[:,:]))
end

function deep_copy(X::NormalToricVariety)
    rays = copy(getRays(X))
    cones = copy(getCones(X))
    Y = normal_toric_variety(IncidenceMatrix(cones), reduce(hcat,rays))
    return Y
end

function getConeFaces(fan::NormalToricVariety,cone::AbstractVector)
    faces = getCones(fan)
    rayMatrix = getRays(fan)
    cone_faces=[]
    c = rank(matrix(QQ,reduce(hcat,newRowMinors(rayMatrix,cone))[:,:])) - 1
    rank_c_subcones = Array{Array{Int64}}([])
    for subcone in faces
        d = rank(matrix(QQ,reduce(hcat,newRowMinors(rayMatrix,subcone))[:,:]))
        if d == c
            push!(rank_c_subcones, subcone)
        end
    end
    for subcone in rank_c_subcones
        if all((i -> i in cone).(subcone))
            push!(cone_faces, subcone)
        end
    end 
    return cone_faces
end

function convertIncidenceMatrix(A::Polymake.IncidenceMatrixAllocated{Polymake.NonSymmetric})
    A=Array(A)
    dim1=size(A,1)
    dim2=size(A,2)
    out=[]
    for i in 1:dim1
        members=[]
        for j in 1:dim2
            if A[i,j]==true
                append!(members,j)
            end
        end
        append!(out,[members])
    end
    return convert(Array{Array{Int64,1},1}, out)
end

# function listToMatrix(L::Union{Vector{Array{QQFieldElem}}, Vector{Vector{QQFieldElem}}})
#     colSize = size(L[1],1)
#     rowSize = size(L,1)
#     return copy(reshape(transpose(collect(Iterators.flatten(L)), rowSize,colSize )))
# end

function listToMatrix(L::Union{Vector{Array{QQFieldElem}}, Vector{Vector{QQFieldElem}}})
    Lmod =map(x -> copy(reshape(collect(Iterators.flatten([x])), size([x],1),size([x][1],1) )),L)
    mat = Lmod[1]
    for i in 2:size(Lmod,1)
        mat = vcat([mat,Lmod[i]]...)
    end
    return mat
end


function newListToMatrix(L::Union{Vector{Array{QQFieldElem}}, Vector{Vector{QQFieldElem}}})
    colSize = size(L[1],1)
    rowSize = size(L,1)
    return copy(reshape( collect(Iterators.flatten(L)), rowSize,colSize ))
end


function new_toric_blowup(s, X, v)
    #If v is not provided, blow up X at the barycenter of s.
    if v==nothing
        v=findBarycenter(s,X)
    end
    if size(s,1)==1
        return X
    end
    #
    coneList = Array{Array{Int64}}(getMaximalCones(X))
    subcones = getCones(X)
    starIndex = findall((t) -> all(((i) -> i in t).(s)), coneList)
    star = [coneList[i] for i in starIndex]

    
    rayMatrix = getRays(X)
    
    clStar = []

    for t in star
        c = rank(matrix(QQ,reduce(hcat,newRowMinors(rayMatrix,t))[:,:])) - 1
        rank_c_subcones = Array{Array{Int64}}([])
        for subcone in subcones
            d = rank(matrix(QQ,reduce(hcat,newRowMinors(rayMatrix,subcone))[:,:]))
            if d == c
                push!(rank_c_subcones, subcone)
            end
        end
        for cone in rank_c_subcones
            if all((i -> i in t).(cone))
                push!(clStar, cone)
            end
        end
    end

    
    
    clStar = unique(clStar)

    n = size(rayMatrix, 1) + 1
    coneList = filter(x -> !(x in star), coneList)
    if length(s) == 1
        # If s consists of a single ray, find all the cones in clStar that does not contain s
        newCones = []
        for t in clStar
            if !(s[1] in t)
                push!(newCones, sort(push!(t, s[1])))
            end
        end
        # return newCones plus coneList
        finalCones = [[i for i in cone] for cone in append!(coneList, newCones)]
        finalRays = reduce(hcat, getRays(X))
        return normal_toric_variety(IncidenceMatrix(finalCones), Array(finalRays))
    end
    newCones = []
    for t in clStar
        # Find all the cones in clStar that does not contain at least one ray in s
        # QUESTION: Why seperate this from the one element case? Any won't work with one element list?
        if any(((i) -> !(i in t)).(s))
            push!(newCones, push!(t, n))
        end
    end
    # return newCones plus coneList
    finalRaysList = append!(getRays(X), [[QQ(i) for i in v]])
    finalCones = [[i for i in cone] for cone in append!(coneList, newCones)]
    rowLength = size(finalRaysList[1],1)
    colLength = size(finalRaysList,1)
    transposeFinalRaysList = [[finalRaysList[j][i] for j in [1:1:colLength;]] for i in [1:1:rowLength;]]
    transposeFinalRays = reduce(hcat, transposeFinalRaysList)
    return normal_toric_variety(IncidenceMatrix(finalCones), transposeFinalRays)
end



"""
    stackyBlowup(::StackyFan,::Array{Int64,1},::Array{Int64,1})

    Takes a stacky fan sf, a ray excep, and a cone, and subdivides the stacky fan at the given ray. Crucially, the given cone should be the minimal cone containing the exceptional ray. The cone input should be zero-indexed.

#examples
```jldoctest
julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 1 2],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[2,3]);

julia> stackyWeights(stackyBlowup(F,[0,1],[1,1]))
[ 2 ,  1 ,  3 ]
```
"""
function stackyBlowup(sf::StackyFan, cone::Array{Int64,1}, excep)
    # Express the exceptional ray as a scalar multiple of a primitive ray
    # Use this scalar as the stacky weight in the resulting stacky fan
    if excep == nothing
        excep = Array{QQFieldElem}(Polymake.common.primitive(findStackyBarycenter(cone,sf)))
    end
    G=gcd(excep)
    excepPM=Polymake.common.primitive(excep)
    excep = [excepPM[i] for i in [1:1:size(excepPM,1);]]
    
    # Perform toric blowup at the given ray
    blowup = new_toric_blowup(cone, sf.fan, excep)
    sf.stacks[Vector{QQFieldElem}(excep)] = G
    return(StackyFan(blowup, sf.stacks))
end


function makeSimplicial(X::NormalToricVariety)
    Y = deepcopy(X) # this should be a copy of X, not X itself
    while (true)
        # If the initial toric variety is simplicial, the program terminates and returns it.
        if is_simplicial(Y)==true
            break
        end
        #Maximal cones and ray matrix
        coneList = getMaximalCones(Y)
        rayMatrix = getRays(Y)
        badCone = nothing
        for i in 1:size(coneList,1)
            cone = coneList[i]
            if (getConeRank(cone, rayMatrix) != size(cone)[1])
                badCone = cone
            end
        end
        if (badCone == nothing)
            # All cones are linearly independent
            break
        else
            # Find the first ray that is contained in more than one orbit
            # and subdivide at that ray, using toricBlowup
            
            # Get faces (need to replace this)
            edges = getConeFaces(Y,badCone)
            # Find the first ray that is contained in more than one orbit
            i = 1
            while count(r->(badCone[i] in r), edges) == 1
                i += 1
            end
            # Subdivide at the cone containing just that ray
            Y = new_toric_blowup(badCone, Y,nothing)
            #Y = toric_blowup([badCone[i]], Y,nothing)
        end
        # Repeat this process until there are no more bad cones
    end
    return Y
end

function makeSmooth(X::NormalToricVariety)
    Y  = deepcopy(X) # this should be a copy
    while(true)
        coneList = fixSubObjects(maximal_cones(Y))
        k = 1
        # Iterate through the coneList, getting the index of the first cone not smooth
        for coneSet in coneList
            # Checking whether this cone is smooth
            smoothCheck=is_smooth(affine_normal_toric_variety(coneSet))
            if !smoothCheck
                # If the cone is not simplicial or not smooth, we have found the cone that we need to make smooth
                break
            else
                k+=1
            end
        end
        # At this point, all the cones are smooth. The program terminates.
        if k == size(coneList,1)+1
            break
        end
        
        # Get the cone that we found to be not smooth
        rayMatrix = getRays(Y)
        sigma = getMaximalCones(Y)[k]
        sigmaRays=newRowMinors(rayMatrix,sigma)

        
        tau=0; tauRays=0; tauCone=0
        # Iterate over the subcones of sigma, finding tau, the smallest one that is not smooth
        for subset in collect(powerset(sigma))
            if size(subset,1) > 1
                subsetRays=newRowMinors(rayMatrix,subset)
                subsetCone=affine_normal_toric_variety(positive_hull(listToMatrix(subsetRays)))
                smoothCheck=is_smooth(subsetCone)
                if !smoothCheck
                    tau=subset
                    tauRays=subsetRays
                    tauCone=subsetCone
                    break
                end 
            end
        end
        # # Getting the Hilbert Basis of tau
        h = hilbert_basis(tauCone)
        H=[map(x -> Integer(x), h[i,:]) for i in [1:1:size(h,1);]]
        rayIndex=0
        # Iterate over the Hilbert Basis, finding the first ray that is not the generator of sigma
        for i in 1:size(H,1)
            if !(H[i] in sigmaRays)
                rayIndex=i
                break
            end
        end
        if rayIndex==0
            # Every Hilbert Basis of tau is a generator of sigma. Make Y simplicial is sufficient to make sigma smooth
            Y=makeSimplicial(Y)
        else
            # blowupRay is not a generator of sigma, blow up tau at blowupRay
            blowupRay=H[rayIndex]
            Y=new_toric_blowup(tau,Y,blowupRay)
        end
        return Y
    end
    return Y
end


"""
    remove!(::Array{Int64},::Int64)

    In-place removes a given item from a vector.

# Examples
```jldoctest
julia> A=[1,2,3,4];

julia> remove!(A,1);

julia> A
[2,3,4]
```
"""
function remove!(a::Array{Int64,1}, item::Int64)
    return deleteat!(a, findall(x->x==item, a))
end
"""
    getIndex(::Array{Int64,1},::Array{Int64,2})

    Returns the first index at which a vector appears as a row of a matrix.

# Examples
```jldoctestx
getIndex([0,1,0],[1 0 0; 0 1 0; 0 0 1])
2
``` 
"""
function getIndex(ray::Array{QQFieldElem,1},rayMatrix::Array{QQFieldElem,2})
    slice=slicematrix(rayMatrix)
    index=findfirst(x->x==ray,slice)
    return index
end
    
"""
    isIndependent(::Int64,::Array{Int64,1},::Array{Int64,2})

    Takes a ray matrix, a list of indices representing a cone, and an index represeting a ray of that cone. Determines whether the given ray is independent in the cone (i.e. does not contribute to the multiplicity of the cone).

# Examples
```jldoctest
julia> isIndependent(3,[1,2,3],[1 0 0; 0 1 0; 1 2 3])
false

julia> isIndependent(3,[1,2,3],[1 0 0; 0 1 0; 1 1 1])
true
```
"""
function isIndependent(rayIndex::Int64,cone::Array{Int64,1},rayMatrix::Array{QQFieldElem,2})
    if size(cone,1)==1
        return true
    end
    scone=copy(cone)
    subcone=remove!(scone,rayIndex)
    mult=getMultiplicity(cone,rayMatrix)
    submult=getMultiplicity(subcone,rayMatrix)
    return mult==submult
end
    
"""
    independencyIndex(::Array{Int64,1},::Array{Int64,2})

    Returns the number of non-independent rays in a cone. Input in indices-ray matrix format.

# Examples
```jldoctest
julia> independencyIndex([1,2,3],[1 0 0 ; 1 2 0; 2 0 3; 0 0 5])
2
```
"""
function independencyIndex(cone::Array{Int64,1},rayMatrix::Array{QQFieldElem,2})
    return count(elt -> !isIndependent(elt, cone, rayMatrix), cone)
end  


# Examples
"""
```jldoctest
julia> F=makeStackyFan([1 0 0; 1 2 0; 0 0 1],[[0,1,2]],[1,1,2]);

julia> isRelevant([1,2,0],[1,2,3],F)
true

julia> F=makeStackyFan([1 0 0; 0 1 0; 0 0 1],[[0,1,2]],[1,1,2]);

julia> isRelevant([0,1,0],[1,2,3],F)
false
```
"""

function isRelevant(ray::Array{QQFieldElem,1},cone::Array{Int64,1},F::StackyFan)
    rayMatrix = reduce(hcat,(myTranspose(getRays(F.fan))[:,:]))
    rayStack=F.stacks[ray]
    rayIndex=findall(x->x==ray,getRays(F.fan))[1]
    rayIndependent=isIndependent(rayIndex,cone,rayMatrix)
    return rayStack != 1 || !rayIndependent

    
end

"""
    toroidalIndex(::Array{Int64,1},::StackyFan,::Dict)
    
    Calculates the toroidal index of the given cone of a divisorial stacky fan, or the number of relevant non-divisorial rays. Compare to divisorialIndex().
    
# Examples
```jldoctest
julia> F=makeStackyFan([1 0 0; 0 1 0; 1 0 2],[[0,1,2]],[1,2,1]);

julia> div=Dict([1,0,0]=>0,[0,1,0]=>0,[1,0,2]=>0);
    
julia> toroidalIndex([1,2,3],F,div)
3
julia> div=Dict([1,0,0]=>0,[0,1,0]=>0,[1,0,2]=>1);
    
julia> toroidalIndex([1,2,3],F,div)
2
```
"""
function toroidalIndex(cone::Array{Int64,1},F::StackyFan,div::Dict)
    slice = getRays(F.fan)
    # Find number of divisorial rays 
    # div is a dictionary that represents which rays are divisorial, 0 represents a non-divisorial ray and 1 represents a divisorial ray
    s=count(x->div[slice[x]]==1,cone)
    #flipt counts the number of non-divisorial and irrelevant rays
    flipt = count(i -> div[slice[i]]==0 && !isRelevant(slice[i], cone, F), cone)
    # number of relevant cones
    t=size(cone,1)-flipt
    # return the toroidal index, which is the number of relevant residual (non-divisorial) rays
    return t-s
end




"""
    divisorialIndex(::Array{Int64,1},::StackyFan,::Dict)

    Calculates the divisorial index (defined by Daniel Bergh) of a given cone in a fan with divisorial rays. 
    Specifically, takes the subcone consisting of all relevant non-divisorial rays in a cone, and counts the number of rays that are relevant in that subcone.

# Examples
```jldoctest
julia> F=makeStackyFan([1 0 0; 0 1 0; 1 0 2],[[0,1,2]],[1,2,1]);

julia> div=Dict([1,0,0]=>0,[0,1,0]=>0,[1,0,2]=>0);
    
julia> divisorialIndex([1,2,3],F,div)
3
julia> div=Dict([1,0,0]=>0,[0,1,0]=>0,[1,0,2]=>1);
    
julia> divisorialIndex([1,2,3],F,div)
1
```
"""
function divisorialIndex(subcone::Array{Int64,1},F::StackyFan,div::Dict{Vector{QQFieldElem}, Int64})
    slicedRayMatrix= getRays(F.fan)
    relRes=Vector{QQFieldElem}[]
    relResStack=QQFieldElem[]
    # c is the number of non-divisorial relevant cones
    c=0
    for i in subcone
        ray=slicedRayMatrix[i]
        stack=F.stacks[ray]
        # If the ray is non-divisorial and relevant, increment c by one, add ray to relRes, and stack to relResStack. The rays in relRes are used to build a new cone.
        if div[ray]==0 && isRelevant(ray,subcone,F)
            c+=1
            push!(relRes,ray)
            push!(relResStack,stack)
        end
    end
    # If there are no relevant residual cones in F, the divisorial index is 0
    if c==0
        return 0
    else
        # convert to 0-indexing
        relResIndz=[[i for i in 1:c]]
        relResInd=[i for i in 1:c]
        relResCat=copy(transpose(hcat(relRes...)))
        # Construct a subfan consisting of all relevant residual rays
        subfan= normal_toric_variety(IncidenceMatrix(relResIndz),relResCat)
        pairs = map((x,y) -> (x,y), slicematrix(relResCat), relResStack)
        stacks = Dict(pairs)
        substackyfan=StackyFan(subfan,stacks)
        divInd=0
        for ray in relRes
            # Iterate over relevant residual cones to count the number of relevant rays in the subfan
            # This count represents the divisorial index
            if isRelevant(ray,relResInd,substackyfan)
                divInd+=1
            end
        end
        return divInd
        return relRes
    end
end

"""
    coneRayDecomposition(::Array{Int64,1},::Array{Int64,2},::Array{Int64,1},::Array{Int64,1})

    This function takes in a cone (a vector of indices of cone generators in rayMatrix), a ray, and a stacky structure for rayMatrix. It first multiplies all generators of the cone by their stacky values, and then finds an expression for the ray as a sum of these stacky generators. The output is a vector of coefficients of the above representation in terms of the rays in rayMatrix, with zeros as coefficients for all rays not in the given cone.

# Examples
```jldoctest
julia> coneRayDecomposition([1,2,3],[3 5 7; 8 16 9;2 1 3;1 1 1],[2,2,3],[1,1,1,1])
[ 6 ,  5 ,  52 ,  0 ]
```
"""
function coneRayDecomposition(cone,rayMatrix,ray,stack)
    stackMatrix=diagm(stack)*rayMatrix # multiply all rays by stack values
    coneRays=rowMinors(stackMatrix,cone) # find the (stackified) rays in the given cone
    if rank(coneRays)<size(coneRays,1)
        error("The given cone is not simplicial.")
    end
    B=Polymake.common.null_space(hcat(transpose(coneRays),-ray)) # Express the input ray in terms of the stackified cone generators
    N=convert(Array{Int64,1},vec(B))
    if size(N,1)==0
        error("The given ray is not in the span of the cone generators.")
    end
    if N[end]<0 #since the nullspace has arbitrary sign, fix it so the coefficients are all positive
        N*=-1
    end
    pop!(N)
    out=zeros(Int64,size(rayMatrix,1)) 
    for i in 1:size(N,1)#rewrite the coefficients vector in terms of all the rays in rayMatrix, by padding with zeros when appropriate.
        out[cone[i]]=N[i] 
    end
    return out
end


"""
    interiorPoints(::Polymake.BigObjectAllocated)

    Finds all interior lattice points contained in the fundamental region of a given cone. When multiple interior lattice points lie along the same ray, only the point closest to the origin is returned. Notably, 

# Examples
```jldoctest
julia> C=Polymake.polytope.Cone(INPUT_RAYS=[1 2; 2 1]);

julia> interiorPoints(C)
[[ 1 ,  1 ]]
```
"""
function interiorPoints(C::Polymake.BigObjectAllocated)
    rayMatrix=Array(Polymake.common.primitive(C.RAYS))
    l=size(rayMatrix,1)
    dim=size(rayMatrix,2)
    if rank(rayMatrix)<l
        print(rayMatrix)
        error("Input cone is not simplicial.")
    end
    subsets=collect(powerset([1:l;]))
    vertices=[]
    for elt in subsets #vertices of the fundamental region are in correspondence with subsets of the generators of the cone, by summing the generators in a subset to obtain a vertex
        vert=zeros(Polymake.Rational,1,dim)
        for i in 1:l
            if i in elt
                vert+=rayMatrix[[i],:]
            end
        end
        append!(vertices,[vert])
    end
    V=vcat(vertices...)
    VH=hcat(ones(Polymake.Rational,size(V,1)),V)
    P=Polymake.polytope.Polytope(POINTS=VH) #make a Polymake polytope object from the vertices of the fundamental region found in the last step
    if size(P.INTERIOR_LATTICE_POINTS,1)==0
        return nothing
    end
    intPoints=Array(P.INTERIOR_LATTICE_POINTS)[:,2:(dim+1)] #find all the interior lattice points
    validPoints=[]
    #return intPoints
    for i in 1:size(intPoints,1) #throw out all points that are integer multiples of other points
        point=intPoints[i,:]
        if gcd(point)==1
            append!(validPoints,[point])
        end
    end
    return validPoints
end


"""
    coneMultiplicity(C::Polymake.BigObjectAllocated)

    Returns the multiplicity of a polyhedral cone (inputted as a Polymake object): here, the multiplicity is defined as the index of the sublattice generated by the rays of the cone, inside the full integer lattice contained in the linear subspace generated by the edges of the cone.

# Examples
```jldoctest

julia> C=Polymake.polytope.Cone(INPUT_RAYS=[1 0; 1 2]);

julia> coneMultiplicity(C)
2
```
"""
function coneMultiplicity(C::Polymake.BigObjectAllocated)
    A=Polymake.common.primitive(C.RAYS)
    M=matrix(ZZ,[ZZ.(y) for y in A])
    SNF=Nemo.snf(M)
    mult=1
    for i in 1:size(SNF,1)
        mult*=SNF[i,i]
    end
    return mult
end

"""
    getMultiplicity(::Array{Int64,1},::Array{Int64,2})
        
    Same functionality as coneMultiplicity, but calculates the cone rays as a subset of the columns of a ray matrix rather than from a Polymake cone object.
        
# Examples
```jldoctest
julia> getMultiplicity([1,2],[1 0; 1 2; 1 3])
2       
```     
"""
function getMultiplicity(cone::Array{Int64,1},rayMatrix)
    A=rowMinors(rayMatrix,cone)
    M=matrix(ZZ,[ZZ.(y) for y in A])
    SNF=Nemo.snf(M)
    mult=1
    for i in 1:size(SNF,1)
        mult*=SNF[i,i]
    end
    return mult
end

"""
    coneConvert(::abstractVector{Int64},::abstractMatrix{Int64})

    Takes a matrix where the columns represent rays, and a list of indices, and forms a Polymake cone object generated by the rays corresponding to those indices.

# Examples
```jldoctest

julia> typeof(coneConvert([1, 2, 4],[1 0 0; 0 1 0; 0 0 1; 1 1 1]))
Polymake.BigObjectAllocated
```
"""
function coneConvert(cone::Array{Int64,1},rayMatrix::Array{Int64,2})
    coneRays=rowMinors(rayMatrix,cone)
    C=Polymake.polytope.Cone(RAYS=coneRays)
    return C
end
    

"""
    convertToIncidence(v::Array{Int64,1},l::Int64)

Returns a vector of length l, with entries of 1 indexed by v and entries of 0 everywhere else.

# Examples
```jldoctest
julia> convertToIncidence([2,3,5],6)
[ 0 , 1 , 1 , 0 , 1 , 0 ]
```
"""
function convertToIncidence(v::Array{Int64,1},l::Int64)
    out=[]
    for j in 1:l
        if j in v
            append!(out,1)
        else
            append!(out,0)
        end
    end
    return out
end

"""
    convertBool(::AbstractVector)

Takes a column vector of boolean values and converts it to a vector of indices marked 'true'.

#Examples
```jldoctest makeSmoothWithDependencies
julia> B=[true,true,false,true]

julia> convertBool(B)
[0, 1, 3]
```
"""
function convertBool(B::AbstractVector)
    out=Int64[]
    for i in 1:size(B,1)
        if B[i]==true
           append!(out,i-1) 
        end
    end
    return out
end

"""
    compareCones(::Array{Int64,1},::Array{Int64,1},::Array{Int64,2},::Array{Int64,1})

    Takes in two cones (in index vector notation), a ray matrix, and a incidence vector of distinguished rays. If the cones do not have an equal number of distinguished rays, returns the difference between the two values. Otherwise, returns the difference in the cone multiplicities.

# Examples
```jldoctest AlgA
julia> compareCones([1,2],[2,3],[1 0 0; 0 1 0; 0 0 1],[1,1,0])
1

julia> compareCones([1,2],[1,3],[1 0;1 2;1 -1],[1,1,1])
1
```
"""
function compareCones(cone1::Array{Int64,1}, cone2::Array{Int64,1}, rayMatrix::Array{Int64,2}, distinguished::Array{Int64,1})
    l=size(rayMatrix,1)
    c1=convertToIncidence(cone1,l)
    c2=convertToIncidence(cone2,l)
    # Calculate the number of non-distinguished rays
    nondist1 = size(cone1,1) - dot(c1, distinguished)
    nondist2 = size(cone2,1) - dot(c2, distinguished)
    if (nondist1 - nondist2 != 0)
        return nondist1 - nondist2
    else
        # Need to use the method for calculating multiplicity of cone
        mult1 = coneMultiplicity(coneConvert(cone1,rayMatrix))
        mult2 = coneMultiplicity(coneConvert(cone2,rayMatrix))
        return mult1 - mult2
    end
end

"""
    extremalCones(::Array{Array{Int64,1},1},::Array{Int64,2},::Array{Int64,1})

    Takes a list of vectors representing cones in a fan, a ray matrix, and a vector representing the distinguished rays as 0 or 1 values, and calculates the cones that are maximal with respect to (first) the number of non-distinguished rays and (second) the multiplicity of the cone. In Bergh's algorithm A (where this ordering is used), the input S will consist only of those cones containing at least one distinguished ray and at least one interior point.

#Examples
```jldoctest AlgA
julia> extremalCones([[1,2],[2,3],[3,4]],[1 0;1 2; 1 5; 1 8],[0,1,1,0])
[[ 3 ,  4 ]]
```
"""
function extremalCones(S::Vector{Vector{Int64}}, rayMatrix::Array{Int64,2}, distinguished::Array{Int64,1})
    # Finds the extremal cones according to # distinguished rays and multiplicity
    # distinguished is a boolean vector whose size is equal to the number of rays
    # The i-th index is 1 if the i-th ray (in rayMatrix) is distinguished
    maxCones = [S[1]]
    for i in 2:size(S,1)
        cone = S[i]
        # Compare the cone with the first element of the maximal cone list
        comp = compareCones(cone, maxCones[1], rayMatrix, distinguished)
        if comp > 0
            maxCones = [cone]
        elseif comp == 0
            push!(maxCones, cone)
        end
    end
    return maxCones
end

"""
    distinguishedAndIntPoint(::Array{Int64,1},::Array{Int64,2},::Array{Int64,1})

    Calculates if the cone formed by a subset of rays in rayMatrix indexed by the entries of cone, and with a distinguished structure given by the incidence vector dist, both contains at least one distinguished ray and contains a proper interior point.

# Examples
```jldoctest AlgA
julia> distinguishedAndMultiplicity([1,2,4],[1 0 0; 1 2 0;2 1 3; 1 0 3],[1,0,0,0])
true
```
"""
function distinguishedAndIntPoint(cone::Array{Int64,1},rayMatrix::Array{Int64,2},dist::Array{Int64,1})
    l=size(rayMatrix,1)
    if dot(convertToIncidence(cone,l),dist) > 0 #check distinguished
        C=coneConvert(cone,rayMatrix)
        if interiorPoints(C)!=nothing #check interior point
            return true
        else
            return false
        end
    else
        return false
    end
end

"""
    minimalByLex(::Array{Array{Int64,1},1})

    Given a list of vectors of equal length, returns the minimal vector with respect to lexicographic ordering.

# Examples
```jldoctest AlgA
julia> A=[[1,1,1],[2,1,3],[0,5,4]];

julia> minimalByLex(A)
[ 0 ,  5 ,  4 ]
```
"""
function minimalByLex(A::Array{Array{QQFieldElem,1},1})
    l=size(A,1)
    minimal=A[1]
    d=size(minimal,1)
    for i in 2:l
        test=A[i]
        for j in 1:d
            if minimal[j]<test[j]
                break
            elseif minimal[j]>test[j]
                minimal=test
                break
            end
        end
    end
    return minimal
end

"""
    minimalByDist(::Array{Array{Int64,1},1},::Array{Int64,1})

    Given a list of vectors (representing rays as weighted sums of other rays) and a vector of 0's and 1's representing non-distinguished and distinguished indices, returns a vector from the list such that the sum of the entries corresponding to distinguished indices is minimized.

#Examples
```jldoctest AlgA
julia> minimalByDist([[0,1,5,7],[3,3,2,2],[8,5,3,6],[2,1,1,10]],[0,1,1,0])
[ 3 , 3 , 2 , 2 ]
```
"""
function minimalByDist(A::Array{Array{Int64,1},1},D::Array{Int64,1})
    l=size(A,1)
    minimal=A[1]
    d=size(minimal,1)
    for i in 2:l
        test=A[i]
        if dot(test,D)<dot(minimal,D)
            minimal=test
        end
    end
    return minimal
end


function convertToInt(A)
    return map(x->Int(numerator(x)), A)
end

"""
  coneContains(::Array{Int64,1},::Array{Int64,1})
    
    Checks whether every index in the first input is also contained in the second input. 
    
# Examples
```jldoctest
julia> coneContains([1,2,3],[1,2,3,4])
true
julia> coneContains([1,2,5],[1,2,3,4])
false
```
"""

function coneContains(A::Array{Int64,1},B::Array{Int64,1})
    return issubset(A, B)
end


"""
    minMaxDivisorial(::StackyFan,::Dict)
    
    Calculates the maximal divisorial index of all cones in a stacky fan. 
    Each maximal cone of the fan will contain at most one minimal subcone of maximal divisorial index; a list of such cones is returned.

# Examples
```jldoctest
julia> F=makeStackyFan([1 2 0;1 3 0; 3 0 1],[[0,1,2]],[1,1,5]);
    
julia> div=Dict([1,2,0]=>0,[1,3,0]=>0,[3,0,1]=>0);
    
julia> minMaxDivisorial(F,div)
[[3]]
    
julia> F=makeStackyFan([1 1 0;1 3 0; 3 0 1],[[0,1,2]],[1,1,5]);

julia> div=Dict([1,1,0]=>0,[1,3,0]=>0,[3,0,1]=>0);
    
julia> minMaxDivisorial(F,div)
[[1,2,3]]
```
""" 
function minMaxDivisorial(F::StackyFan,div::Dict{Vector{QQFieldElem}, Int64})
    # Calculates the maximal divisorial index of any cone in the fan
    divMax=0
    coneList=getCones(F.fan)
    # dictionary that represents each cone with its divisorial index
    divisorialDict=Dict()
    for cone in coneList
        d=divisorialIndex(cone,F,div)
        divisorialDict[cone]=d
        if d>divMax
            divMax=d
        end
    end 
    if divMax==0
        return nothing
    end

    # cones with maximal divisorial index
    divMaxCones=Array{Int64,1}[]
    for cone in coneList
        if divisorialDict[cone]==divMax
             # if the cone's divisorial index is the fan's maximal divisorial index, add the cone to divMaxCones
            push!(divMaxCones,cone)
        end
    end

    #divMaxConesRefined stores the cones in divMaxCones that are minimal with respect to inclusion
    divMaxConesRefined=Array{Int64,1}[]
    # List of maximal cones in F
    maxconeList=getMaximalCones(F.fan)
    for maxcone in maxconeList
        # if the div index of the current maxcone is the fan's max div index, its minimal subcone with maximal divisorial index is calculated
        if divisorialDict[maxcone]==divMax
            maxconeContains=Array{Int64,1}[]
            mincone=maxcone
            for cone in divMaxCones
                if coneContains(cone,maxcone) && size(cone,1)<size(mincone,1)
                    mincone=cone
                end
            end
            if !(mincone in divMaxConesRefined)
                push!(divMaxConesRefined,mincone)
            end
        end
    end
    return divMaxConesRefined
end
