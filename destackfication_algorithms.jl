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
function extremalCones(S::Vector{Array{Int64}}, rayMatrix::Array{Int64,2}, distinguished::Array{Int64,1})
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
    BerghA(F::StackyFan,D::Array{Int64,1})

Given a stacky fan F and a vector of booleans D representing the distinguished structure, returns a smooth stacky fan where the distinguished rays are independent.

The algorithm is adapted from Daniel Bergh's [paper on destackification](https://arxiv.org/abs/1409.5713). In brief, it identifies non-smooth cones containing at least one distinguished ray, finds interior points in those cones, and subdivides at those points through a series of stacky barycentric subdivisions.

# Examples
```jldoctest AlgA

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 2 5],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,1]))
[ 5 ,  2 ,  5 ,  10 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 2 5],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,0]));
[ 5 ,  5 ,  1 ,  2 ,  5 ,  10 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 2 5],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[0,1]))
[ 1 ,  5 ,  5 ,  2 ,  5 ,  1 ,  5 ,  10 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[4 1; 7 9],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,0]))
[ 609 ,  29 ,  1 ,  174 ,  29 ,  1740 ,  1218 ,  58 ,  145 ,  1044 ,  1044 ,  290 ,  290 ,  145 ,  406 ,  348 ,  261 ,  14616 ,  14616 ,  609 ,  3480 ,  870 ,  609 ,  174 ,  609 ,  9744 ,  14616 ,  1218 ,  58 ,  145 ,  435 ,  725 ,  1305 ,  1740 ,  6960 ,  3480 ,  870 ,  3480 ,  58464 ,  6090 ,  3480 ,  1392 ,  696 ,  1044 ,  2088 ,  261 ,  174 ,  261 ,  609 ,  406 ,  609 ,  609 ,  406 ,  609 ,  1218 ,  812 ,  1218 ,  2088 ,  261 ,  1044 ,  1160 ,  1740 ,  1740 ,  870 ,  1305 ,  1305 ,  14616 ,  10440 ,  145 ,  435 ,  609 ,  116 ,  580 ,  290 ,  580 ,  1740 ,  3480 ,  3480 ,  261 ,  522 ,  261 ,  522 ,  522 ,  1218 ,  1218 ,  1218 ,  1218 ,  2436 ,  2436 ,  4176 ,  2088 ,  3480 ,  2610 ,  29232 ,  6090 ,  1218 ,  290 ,  145 ,  290 ,  12180 ,  261 ,  522 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[323 179; 44 135],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,1]))
[ 491602456800 ,  49160245680 ,  294961474080 ,  468192816 ,  12173013216 ,  73038079296 ,  12173013216 ,  97384105728 ,  73038079296 ,  196640982720 ,  131093988480 ,  156064272 ,  936385632 ,  468192816 ,  196640982720 ,  49160245680 ,  9832049136 ,  292152317184 ,  5899229481600 ,  3932819654400 ,  589922948160 ,  393281965440 ,  12173013216 ,  8115342144 ,  12173013216 ,  5899229481600 ,  589922948160 ,  737403685200 ,  51126655507200 ,  292152317184 ,  51126655507200 ,  5899229481600 ,  5899229481600 ,  589922948160 ,  589922948160 ,  196640982720 ,  2949614740800 ,  5899229481600 ,  294961474080 ,  589922948160 ,  196640982720 ,  589922948160 ,  393281965440 ,  936385632 ,  24346026432 ,  24346026432 ,  11798458963200 ,  1179845896320 ,  737403685200 ,  1474807370400 ,  2949614740800 ,  5899229481600 ,  589922948160 ,  11798458963200 ,  1179845896320 ,  2949614740800 ,  1474807370400 ,  5899229481600 ,  102253311014400 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0 3; 4 5 6; 2 3 1],INPUT_CONES=[[0,1,2]]);

julia> F=addStackStructure(X,[1,1,1]);

julia> stackyWeights(BerghA(F,[1,1,1]))
[ 28 ,  21 ,  84 ,  28 ,  84 ,  84 ,  42 ,  84 ,  168 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0 2; 2 1 1; 5 3 9],INPUT_CONES=[[0,1,2]]);

julia> F=addStackStructure(X,[1,1,1]);

julia> stackyWeights(BerghA(F,[1,1,1]))
[ 4 ,  4 ,  8 ,  4 ,  4 ,  8 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 4 3; 1 5],INPUT_CONES=[[0,1],[1,2]]);

julia> F=addStackStructure(X,[1,1,1]);

julia> stackyWeights(BerghA(F,[1,1,1]))
[ 4 ,  34 ,  6 ,  68 ,  34 ,  68 ,  34 ,  6 ,  12 ]
```
"""
function BerghA(F::StackyFan,D::Array{Int64,1};verbose::Bool=false)
    if verbose==true
        println("==algorithm is running in verbose mode==")
        println(" ")
        println("=======")
    end
    X=deepcopy(F)
    
    #rayMatrix=convert(Array{Int64,2},Array(Polymake.common.primitive(X.fan.RAYS)))
    rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
    rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
    
    coneList=getCones(X.fan)
    dim=size(rayMatrix,2)
    numRays=size(rayMatrix,1)
    
    #check if the vector D has length equal to the number of rays in F
    if numRays != size(D,1)
        error("length of vector representing distinguished structure does not agree with number of rays in stacky fan.")
    end
    
    #A0: initialization
    i=0
    while(true)
        rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
        rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
        
        numRays=size(rayMatrix,1)
        coneList=getCones(X.fan)
        
        #A1: Find S the set of cones that contain a distinguised ray and an interior lattice point 
        #Note: cones in S are 1-indexed.
        S=filter(cone->distinguishedAndIntPoint(cone,rayMatrix,D),coneList)
        # If S is empty, the program terminates.
        if S==[]
            break
        end
        
        #A2 - find extremal cones
        Smax=extremalCones(S,rayMatrix,D)
        Smax = map(x->sort(x),Smax)
        
        #Print information on the number of extremal cones, their number of non-distinguished rays, and their multiplicity
        #The algorithm is structured to first reduce the number of non-distinguished rays in extremal cones, and then reduce the multiplicity of said cones,
            #so this information can be used to track the algorithm's progress
        if verbose==true
            Smaxcount=size(Smax,1)
            println("Number of extremal cones: $Smaxcount")
            testCone=Smax[1]
            c1=convertToIncidence(testCone,numRays)
            nonDist=size(testCone,1)-dot(c1,D)
            mult=coneMultiplicity(coneConvert(testCone,rayMatrix))
            println("Maximal non-distinguished rays and multiplicity: $nonDist, $mult")
        end
        
        #A2 - find interior points in Smax
        intPoints=[]
        for cone in Smax
            #C=rowMinors(rayMatrix,cone)
            C=coneConvert(cone,rayMatrix)
            coneIntPoints=interiorPoints(C)
            for point in coneIntPoints
               push!(intPoints,(point,cone)) #the point is stored as a tuple along with the cone containing it
            end
        end
        
        #A2 - find stacky points (in terms of coefficients) derived from interior points
        P=Array{Int64,1}[]
        for (point,cone) in intPoints
            stackyWeight = Array{Int64}(Polymake.common.primitive(stackyWeights(X)))
            stackyPoint=coneRayDecomposition(cone,rayMatrix,point,stackyWeight) #each interior point is rewritten as a string of coefficients
                #corresponding to its representation as a sum of stacky rays
            push!(P,stackyPoint)
        end
        

        
        #A2 - find element of P such that the sum of the coefficients corresponding to distinguished rays is minimal.
            #This invariant does not produce a unique ray, so there is a degree of arbitrary selection.
        psi=minimalByDist(P,D)
        #if verbose==true
            #println("Psi: $psi")
        #end
        
        #A3 - perform root construction
        X=rootConstructionDistinguishedIndices(X,D,psi)

        

        #A3 - modify psi with respect to root construction
        for i in 1:length(psi)
            if D[i]==1 && psi[i]>1
                psi[i]=1
            end
        end
        
        # A5 - perform repeated stacky barycentric star subdivision with respect to psi.
        while(count(x->x>0,psi)>1)
            
            supportCone=findall(x->x!=0,psi)
            #find the stacky barycenter of that cone, which becomes the exceptional (blowup) ray
            exceptional=findStackyBarycenter(supportCone,X)
            code_rays =  map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            # Track the indices of distinguished rays
            D_pairs = map((x,y) -> (x,y), code_rays, D)
            D_Dict = Dict(D_pairs)
            # Track psi as a linear combination of the generators
            psiPairs = map((x,y) -> (x,y), code_rays,psi)
            psiDict = Dict(psiPairs)
            
            X= stackyBlowup(X,supportCone,exceptional)


            G=gcd(exceptional) #since the blowup ray may not be primitive, it is made primitive and then assigned a stacky value so its stacky form is unchanged.
            primExcep=Array{QQFieldElem}(Polymake.common.primitive(exceptional))
                        
            D_Dict[primExcep]=1
            psiDict[primExcep]=1
            
            newRays=map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            newD=Int64[]
            newpsi=Int64[]
            
            for ray in newRays
                E=ray
                excepCode=primExcep
                push!(newD,D_Dict[E])
                #A4 - modify ps
                if E==excepCode
                    push!(newpsi,1)
                elseif psiDict[E]>1
                    push!(newpsi,psiDict[E]-1)
                else
                    push!(newpsi,0)
                end
            end
            psi=newpsi
            D=newD
        end
        if verbose==true
            println("=======")
        end
        i+=1
    end
    if verbose==true
        println("Number of steps: $i")
    end
    return X,D
end

function BerghAmod(F::StackyFan,D::Array{Int64,1};verbose::Bool=false)
    if verbose==true
        println("==algorithm is running in verbose mode==")
        println(" ")
        println("=======")
    end
    X=deepcopy(F)
    
    #rayMatrix=convert(Array{Int64,2},Array(Polymake.common.primitive(X.fan.RAYS)))
    rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
    rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
    
    coneList=getCones(X.fan)
    dim=size(rayMatrix,2)
    numRays=size(rayMatrix,1)
    
    #check if the vector D has length equal to the number of rays in F
    if numRays != size(D,1)
        error("length of vector representing distinguished structure does not agree with number of rays in stacky fan.")
    end
    
    #A0: initialization
    i=0
    stackyModifications = []
    while(true)
        rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
        rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
        
        numRays=size(rayMatrix,1)
        coneList=getCones(X.fan)
        
        #A1: Find S the set of cones that contain a distinguised ray and an interior lattice point 
        #Note: cones in S are 1-indexed.
        S=filter(cone->distinguishedAndIntPoint(cone,rayMatrix,D),coneList)
        # If S is empty, the program terminates.
        if S==[]
            break
        end
        
        #A2 - find extremal cones
        Smax=extremalCones(S,rayMatrix,D)
        Smax = map(x->sort(x),Smax)
        
        #Print information on the number of extremal cones, their number of non-distinguished rays, and their multiplicity
        #The algorithm is structured to first reduce the number of non-distinguished rays in extremal cones, and then reduce the multiplicity of said cones,
            #so this information can be used to track the algorithm's progress
        if verbose==true
            Smaxcount=size(Smax,1)
            println("Number of extremal cones: $Smaxcount")
            testCone=Smax[1]
            c1=convertToIncidence(testCone,numRays)
            nonDist=size(testCone,1)-dot(c1,D)
            mult=coneMultiplicity(coneConvert(testCone,rayMatrix))
            println("Maximal non-distinguished rays and multiplicity: $nonDist, $mult")
        end
        
        #A2 - find interior points in Smax
        intPoints=[]
        for cone in Smax
            #C=rowMinors(rayMatrix,cone)
            C=coneConvert(cone,rayMatrix)
            coneIntPoints=interiorPoints(C)
            for point in coneIntPoints
               push!(intPoints,(point,cone)) #the point is stored as a tuple along with the cone containing it
            end
        end
        
        #A2 - find stacky points (in terms of coefficients) derived from interior points
        P=Array{Int64,1}[]
        for (point,cone) in intPoints
            stackyWeight = Array{Int64}(Polymake.common.primitive(stackyWeights(X)))
            stackyPoint=coneRayDecomposition(cone,rayMatrix,point,stackyWeight) #each interior point is rewritten as a string of coefficients
                #corresponding to its representation as a sum of stacky rays
            push!(P,stackyPoint)
        end
        

        
        #A2 - find element of P such that the sum of the coefficients corresponding to distinguished rays is minimal.
            #This invariant does not produce a unique ray, so there is a degree of arbitrary selection.
        psi=minimalByDist(P,D)
        #if verbose==true
            #println("Psi: $psi")
        #end
        
        #A3 - perform root construction
        X=rootConstructionDistinguishedIndices(X,D,psi)

        #########################

        all_rays = getRays(X.fan)
        dist_rays = []
        for i in [1:1:size(all_rays, 1);]
            if D[i]==1
                push!(dist_rays,(all_rays[i],psi[i]))
            end
        end
        pair = ("root",dist_rays)
        push!(stackyModifications, pair)

        ##########################

        #A3 - modify psi with respect to root construction
        for i in 1:length(psi)
            if D[i]==1 && psi[i]>1
                psi[i]=1
            end
        end
        
        # A5 - perform repeated stacky barycentric star subdivision with respect to psi.
        while(count(x->x>0,psi)>1)
            
            supportCone=findall(x->x!=0,psi)
            #find the stacky barycenter of that cone, which becomes the exceptional (blowup) ray
            exceptional=findStackyBarycenter(supportCone,X)
            code_rays =  map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            # Track the indices of distinguished rays
            D_pairs = map((x,y) -> (x,y), code_rays, D)
            D_Dict = Dict(D_pairs)
            # Track psi as a linear combination of the generators
            psiPairs = map((x,y) -> (x,y), code_rays,psi)
            psiDict = Dict(psiPairs)

            ################

            raysOfCone = []
            all_rays = getRays(X.fan)
            for i in supportCone
                push!(raysOfCone,all_rays[i])
            end
            pair = ("blowup",raysOfCone)
            push!(stackyModifications, pair)
            ################
            
            X= stackyBlowup(X,supportCone,exceptional)

            G=gcd(exceptional) #since the blowup ray may not be primitive, it is made primitive and then assigned a stacky value so its stacky form is unchanged.
            primExcep=Array{QQFieldElem}(Polymake.common.primitive(exceptional))
                        
            D_Dict[primExcep]=1
            psiDict[primExcep]=1
            
            newRays=map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            newD=Int64[]
            newpsi=Int64[]
            
            for ray in newRays
                E=ray
                excepCode=primExcep
                push!(newD,D_Dict[E])
                #A4 - modify ps
                if E==excepCode
                    push!(newpsi,1)
                elseif psiDict[E]>1
                    push!(newpsi,psiDict[E]-1)
                else
                    push!(newpsi,0)
                end
            end
            psi=newpsi
            D=newD
        end
        if verbose==true
            println("=======")
        end
        i+=1
    end
    if verbose==true
        println("Number of steps: $i")
    end
    return X,D,stackyModifications
end

function getIndex(L, v)
    i= 1
    while L[i] != v
        i+=1
    end
    return i
end

function manualBerghA(F::StackyFan,D::Array{Int64,1}, manual)
    X = deepcopy(F)
    for instruction in manual
        if instruction[1] == "root"
            rootConstruction = instruction[2]
            all_rays = getRays(X.fan)
            dist_ray_indices = []
            for pair in rootConstruction
                distIndex = getIndex(all_rays,pair[1])
                push!(dist_ray_indices, (distIndex,pair[2]))
            end
            full_indices = [0 for i in [1:1:size(all_rays,1);]]
            for pair in dist_ray_indices
                ind = pair[1]
                c = pair[2]
                full_indices[ind] = c
            end
            X = rootConstructionDistinguishedIndices(X,D,full_indices)
        else
            blowup = instruction[2]
            #find the support cone of the blowup
            all_rays = getRays(X.fan)
            supportCone = Array{Int64}([])
            for Ray in blowup
                ind = getIndex(all_rays,Ray)
                push!(supportCone, ind)
            end

            DistPairs = Dict(map((x,y) -> (x,y), all_rays, D))
            barycenter = findStackyBarycenter(supportCone,X)
            primBarycenter = Array{QQFieldElem}(Polymake.common.primitive(barycenter))
            X = stackyBlowup(X,supportCone,barycenter)
            DistPairs[primBarycenter]= 1
            new_all_rays = getRays(X.fan)
            
            newD = [0 for i in [1:1:size(new_all_rays,1);]]
            for i in [1:1:size(new_all_rays,1);]
                theRay = new_all_rays[i]
                newD[i] = DistPairs[theRay]
            end
            D = newD
        end
    end
    return X,D
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

"""
    BerghC(::StackyFan,::Array{Int64,1})
    
    Takes a stacky fan and a binary array indicating which rays are divisorial, and runs Daniel Bergh's algorithm C. 
    This algorithm performs a series of stacky blowups to reduce the maximal divisorial index of the fan, and returns a fan with a maximal divisorial index of 0.

# Examples
```jldoctest
julia> F=makeStackyFan([1 1 0;1 3 0; 0 0 1],[[0,1,2]],[1,1,5]);
    
julia> H, div = BerghC(F,[0,0,0]);
    
julia> convert(Array{Int64,2},Polymake.common.primitive(H.fan.RAYS))
5×3 Matrix{Int64}:
 1  1  0
 0  0  1
 2  4  5
 1  3  0
 1  2  0

julia> div
Dict{Any, Any} with 5 entries:
  [1, 1, 0] => 0
  [0, 0, 1] => 1
  [2, 4, 5] => 1
  [1, 2, 0] => 1
  [1, 3, 0] => 0

julia> F=makeStackyFan([1 0;1 3; 5 17],[[0,1],[1,2]],[1,1,5]);

julia> H, div = BerghC(F,[0,0,0]);

julia> convert(Array{Int64,2},Polymake.common.primitive(H.fan.RAYS))
5×2 Matrix{Int64}:
  1   0
  2   3
  1   3
 13  44
  5  17
julia> div
Dict{Any, Any} with 5 entries:
  [1, 0]   => 0
  [13, 44] => 1
  [1, 3]   => 0
  [2, 3]   => 1
  [5, 17]  => 1
```
""" 
function BerghC(F::StackyFan,divlist::Array{Int64,1})
    X=deepcopy(F)
    slicedRayMatrix=getRays(X.fan)
    div=Dict{Vector{QQFieldElem}, Int64}()
    # Populate dictionary of divisorial rays (div) from divlist
    for i in 1:size(slicedRayMatrix,1)
        div[slicedRayMatrix[i]]=divlist[i]
    end
    while(true)
        div= deepcopy(div)
        slicedRayMatrix=getRays(X.fan)
        # Find the cones with maximal divisorial index in X
        subdivTargetCones=minMaxDivisorial(X,div)
        # If there arde no such cones, the algorithm terminates
        if subdivTargetCones==nothing
            break
        end
        blowupList=Array{Array{QQFieldElem,1},1}[]
        # Iterate through cones with maximal divisorial index
        for cone in subdivTargetCones
            # Add each cone's ray representation to blowupList
            push!(blowupList,newRowMinors(slicedRayMatrix, cone))
        end
        for raycone in blowupList
            indices=Int64[]
            slicedRayMatrix=getRays(X.fan)
            for ray in raycone
                push!(indices,findall(x->x==ray,slicedRayMatrix)[1])
            end
            # Sort indices in ascending order
            cone=sort(indices)
            if size(cone,1)==1
                div[slicedRayMatrix[cone[1]]]=1
            else
                exceptional=findStackyBarycenter(cone,X)
                # perform the blowup
                X=stackyBlowup(X,cone, exceptional)
                # convert exceptional ray to its primitive form
                div[Vector{QQFieldElem}(Polymake.common.primitive(exceptional))]=1
            end
        end
    end
    return X, div
end

"""
    divAlongDivisor(::Array{Int64,1},::StackyFan,::Dict)

    Calculates the simplified divisorial index along a divisor of a given cone in a fan with divisorial rays. Takes the subcone consisting of all relevant residual rays along with the specified divisor, and counts the number of residual rays that are relevant with respect to that subcone.

# Examples
```jldoctest AlgDp
julia> F=makeStackyFan([1 0 0; 0 1 0; 1 0 2],[[0,1,2]],[1,2,1])

julia> div=Dict([1,0,0]=>0,[0,1,0]=>0,[1,0,2]=>1)
    
julia> divAlongDivisor([1,2,3],F,div,[1,0,2])
2
julia> div=Dict([1,0,0]=>0,[0,1,0]=>1,[1,0,2]=>1) #[1 0 0] [1 0 2] - [1 0 0] has non-zero projection
    
julia> divAlongDivisor([1,2,3],F,div,[1,0,2])
1
julia> div=Dict([1,0,0]=>1,[0,1,0]=>0,[1,0,2]=>1) #[0 1 0] has stacky value >1
    
julia> divAlongDivisor([1,2,3],F,div,[1,0,2])
1
julia> div=Dict([1,0,0]=>0,[0,1,0]=>1,[1,0,2]=>1) #[1 0 0] [0 1 0] - [1 0 0] is independent
    
julia> divAlongDivisor([1,2,3],F,div,[0,1,0])
0
```
"""
function divAlongDivisor(cone::Array{Int64,1}, F::StackyFan, div::Dict, divisor::Array{QQFieldElem,1})
    # This may not be a very clear name, doesn't mention "simplified" or "index"
    X=deepcopy(F)

    slicedRayMatrix = getRays(X.fan)
    rayMatrix = listToMatrix(slicedRayMatrix)
    
    # Construct the cone made up of the residual (non-divisorial) rays and the divisor
    residRays = Array{QQFieldElem, 1}[]
    residStack = QQFieldElem[]
    c = 0
    # Add the divisor (which should be divisorial)
    if div[divisor] == 1
        c += 1
        push!(residRays, divisor)
        push!(residStack, F.stacks[divisor])
    else
        error("The given ray is non-divisorial")
    end     
    # Append the other rays
    for i in cone
        ray = slicedRayMatrix[i]
        stack = F.stacks[ray]
        if div[ray] == 0 && isRelevant(ray, cone, F)
            c += 1
            push!(residRays, ray)
            push!(residStack, stack)
        end
    end
    residConeZ = Array(0:c-1)
    residCone = Array(1:c)
    # Create the stacky fan structure containing only the cone
    residRayMatrix = Array{QQFieldElem}(transpose(hcat(residRays...)))
    presubconeFan = normal_toric_variety(IncidenceMatrix([[i+1 for i in residConeZ]]), residRayMatrix)
    subconeFan = addStackStructure(presubconeFan, residStack)

    # Count the number of residual rays which are dependent in the above cone
    # Skip the first ray, which is the divisor
    divIndex = 0
    for i in 2:c
        ray = residRays[i]
        if isRelevant(ray, residCone, subconeFan)
            divIndex += 1
        end
    end
    return divIndex
end


"""
    positiveDivIndexCone()

    Tests whether there exists a cone with non-zero divisorial index along the given divisor.
"""
function positiveDivIndexCone(divisor::Array{Int64,1}, F::StackyFan, div::Dict)
    # Go through each maximal cone containing the ray, and testing the divisorial index
    rayMatrix = convert(Array{Int64,2}, Array(Polymake.common.primitive(X.fan.RAYS)))
    maxconeList = convertIncidenceMatrix(F.fan.MAXIMAL_CONES)
    divisorIndex = getIndex(divisor, rayMatrix)
    for maxCone in maxconeList
        if divisorIndex in maxCone
            coneDivIndex = divAlongDivisor(maxCone, F, div, divisor)
            # If the divisorial index along the divisor is non-zero, return true
            if coneDivIndex > 0
                return true
            end
        end
    end
    return false
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
function minMaxIndexAlongDiv(F::StackyFan,div::Dict{Vector{QQFieldElem}, Int64}, ray::Vector{QQFieldElem})
    # Calculates the maximal divisorial index of any cone in the fan
    divMax=0
    coneList=getCones(F.fan)
    # dictionary that represents each cone with its divisorial index
    divisorialDict=Dict()
    for cone in coneList
        d=divAlongDivisor(cone,F,div, ray)
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


function positiveDivIndexCone(divisor::Array{QQFieldElem,1}, F::StackyFan, div::Dict)
    X = deepcopy(F)
    while(true)
        rayMatrix = listToMatrix(getRays(X.fan))
        maxconeList = getMaximalCones(X.fan)
        divisorIndex = getIndex(divisor, rayMatrix)
        for maxCone in maxconeList
            if divisorIndex in maxCone
                coneDivIndex = divAlongDivisor(maxCone, F, div, divisor)
                # If the divisorial index along the divisor is non-zero, return true
                if coneDivIndex > 0
                    return true
                end
            end
        end
    end
    return false
end


function BerghD(F::StackyFan,divlist::Array{Int64,1}, Dlist::Array{Int64,1})
    X=deepcopy(F)
    slicedRayMatrix=getRays(X.fan)
    div=Dict{Vector{QQFieldElem}, Int64}()
    # Populate dictionary of divisorial rays (div) from divlist
    for i in 1:size(slicedRayMatrix,1)
        div[slicedRayMatrix[i]]=divlist[i]
    end
    divisors = Vector{Vector{QQFieldElem}}([])
    for i in [1:1:size(Dlist,1);]
        if Dlist[i] == 1
            push!(divisors,slicedRayMatrix[i])
        end
    end
    for divisor in divisors
        while(true)
            div= deepcopy(div)
            slicedRayMatrix=getRays(X.fan)
            # Find the cones with maximal divisorial index in X
            subdivTargetCones=minMaxIndexAlongDiv(X,div,divisor)
            # If there arde no such cones, the algorithm terminates
            if subdivTargetCones==nothing
                break
            end
            blowupList=Array{Array{QQFieldElem,1},1}[]
            # Iterate through cones with maximal divisorial index
            for cone in subdivTargetCones
                # Add each cone's ray representation to blowupList
                push!(blowupList,newRowMinors(slicedRayMatrix, cone))
            end
            for raycone in blowupList
                indices=Int64[]
                slicedRayMatrix=getRays(X.fan)
                for ray in raycone
                    push!(indices,findall(x->x==ray,slicedRayMatrix)[1])
                end
                # Sort indices in ascending order
                cone=sort(indices)
                if size(cone,1)==1
                    div[slicedRayMatrix[cone[1]]]=1
                else
                    exceptional=findStackyBarycenter(cone,X)
                    # perform the blowup
                    X=stackyBlowup(X,cone, exceptional)
                    # convert exceptional ray to its primitive form
                    div[Vector{QQFieldElem}(normalize(exceptional))]=1
                end
            end
        end
    end
    return X, div
end








function subStackyFan(F::StackyFan, divisor::Vector{QQFieldElem})
    divisorCones = Vector{Vector{Int64}}([])
    nondivisorCones = Vector{Vector{Int64}}([])
    maximalCones = getMaximalCones(F.fan)
    for cone in maximalCones
        if divisor in newRowMinors(getRays(X.fan), cone)
            push!(divisorCones, cone)
        else
            push!(nondivisorCones, cone)
        end
    end
    newFan = normal_toric_variety(IncidenceMatrix(divisorCones),listToMatrix(getRays(F.fan)))
    newStackyFan = StackyFan(newFan,F.stacks)
    return newStackyFan, nondivisorCones
end

function BerghDmodi(F::StackyFan,divlist::Array{Int64,1}, Dlist::Array{Int64,1})
    X=deepcopy(F)
    slicedRayMatrix=getRays(X.fan)
    div=Dict{Vector{QQFieldElem}, Int64}()
    # Populate dictionary of divisorial rays (div) from divlist
    for i in 1:size(slicedRayMatrix,1)
        div[slicedRayMatrix[i]]=divlist[i]
    end
    divisors = Vector{Vector{QQFieldElem}}([])
    for i in [1:1:size(Dlist,1);]
        if Dlist[i] == 1
            push!(divisors,slicedRayMatrix[i])
        end
    end
    for divisor in divisors
        Xsub, nondivisorCones = subStackyFan(X,divisor)
        Xsub, divlist = BerghC(X,divlist)
        divisorCones = getMaximalCones(Xsub.fan)
        finalCones = map(x -> Vector{Int64}(x), append!(divisorCones, nondivisorCones))
        X = makeStackyFan(listToMatrix(getRays(Xsub.fan)),finalCones,X.stacks)
    end
    return X,divlist
end


# X = normal_toric_variety(IncidenceMatrix([[1,2,3]]),[1 1 0;1 3 0; 0 0 1])
# F = addStackStructure(X,[1,1,5])
# H, divi = BerghD(F,[0,0,1],[0,0,1]);
# divi

# function matrixDistCone(Dlist::Vector{Int64},F::StackyFan)
#     rays = map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(F.fan))
#     n = size(rays,1)
#     totalList = [1:1:n;]
#     nonDList = filter(x-> !(x in Dlist), totalList)
#     mat = Array{Array{QQFieldElem}}([])
#     for i in Dlist
#         push!(mat,rays[i])
#     end
#     for i in nonDList
#         push!(mat, rays[i])
#     end
#     return listToMatrix(mat)
# end

function matrixDistCone(Dlist::Vector{Int64},F::StackyFan)
    rays = map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(F.fan))
    raysSubset = [rays[i] for i in Dlist]
    return listToMatrix(raysSubset)
end


function invAndInt(A::AbstractArray)
    if isinvertible(A)
        if findall(x-> x==false,map(x->isinteger(x),A)) == CartesianIndex{2}[]
            return true
        end
    end
    return false
end


function compareDistCones(cone1::Vector{Int64}, cone2::Vector{Int64}, F::StackyFan,div::Array{Int64})
    Dlist = Array{Int64}([])

    for i in [1:1:size(getRays(F.fan),1);]
        if div[i] == 1
            push!(Dlist, i)
        end
    end

    intersection1 = filter(x->x in Dlist, cone1)
    intersection2 = filter(x-> x in Dlist, cone2)


    matrix1 = matrixDistCone(cone1,F)
    matrix2 = matrixDistCone(cone2,F)
    if isinvertible(matrix1) && isinvertible(matrix2)
        matrixToCheck1= *(matrix1, inv(matrix2))
        matrixToCheck2= *(matrix2, inv(matrix1))
        if !(invAndInt(matrixToCheck1) && invAndInt(matrixToCheck2))
            return false
        else
            for distRayIndex in intersection1
                distRay = getRays(F.fan)[distRayIndex]
                tranposeMatrixToCheck2 = Matrix{QQFieldElem}(Transpose(matrixToCheck2))
                mappedRay = *(tranposeMatrixToCheck2,distRay)
                if !(mappedRay in [getRays(F.fan)[i] for i in intersection2])
                    return false
                end
            end
            if size(intersection1, 1) == size(intersection2, 1)
                return true
            end
            return false
        end
    end
end

function artificialOrder(F::StackyFan,div::Array{Int64})
    maximalCones = getMaximalCones(F.fan)
    maximalConeSorted = [[maximalCones[1]]]
    for i in [1:1:size(maximalCones,1);]
        for j in [1:1:i;]
            if compareDistCones(maximalCones[i],maximalConeSorted[j][1],F,div)
                push!(maximalConesSorted[j],maximalCones[i])
            else
                i += 1
            end
        end
    end
end

function divisorialType(F::StackyFan,div::Dict{Vector{QQFieldElem}, Int64}, cone::Array{Int64})
    return 1
end

function aggregrate(cone::Vector{Int64},F::StackyFan,div::Dict{Vector{QQFieldElem}, Int64})
    return (independencyIndex(cone,listToMatrix(getRays(F.fan))), toroidalIndex(cone, F,div),divisorialType(F,div,cone))
end


function minMaxAggregrate(F::StackyFan,div::Dict{Vector{QQFieldElem}, Int64})
    # Calculates the maximal divisorial index of any cone in the fan
    divMax=(0,0,0)
    coneList=getCones(F.fan)
    # dictionary that represents each cone with its divisorial index
    divisorialDict=Dict()
    for cone in coneList
        d=aggregrate(cone,F,div)
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

#modified berghA to record the steps of the operation

function BerghE(F::StackyFan, divlist::Array{Int64,1})
    X = deepcopy(F)
    slicedRayMatrix = getRays(X.fan)
    maximalCones = getMaximalCones(X.fan)
    div=Dict{Vector{QQFieldElem}, Int64}()
    # Populate dictionary of divisorial rays (div) from divlist
    for i in 1:size(slicedRayMatrix,1)
        div[slicedRayMatrix[i]]=divlist[i]
    end

    
    while(true)
        #find the collection of minimal cones with maximal aggregrate 
        conelist = minMaxAggregrate(F, div)

        # if they equal the set of the maximal cones, we're done
        if conelist == maximalCones
            break
        end

        # otherwise, continue
        # for each cone in the collection

        blowupVectors = []
        for cone in conelist
            ############################### working on the model

            modelRayMatrix = listToMatrix([getRays(X.fan)[i] for i in cone])#listToMatrix(getRays(X.fan))
            modelCone = [i for i in [1:1:size(getRays(X.fan));]]
            modelStacks = [stackyWeights(X)[i] for i in cone]
            modelX = normal_toric_variety(IncidenceMatrix([modelCone]),modelRayMatrix)
            modelSX = addStackStructure(modelX,modelStacks)

            
            modelBlowupVector = findStackyBarycenter(cone, X)
            modelBlowupVectorPrim = Array{QQFieldElem}(Polymake.common.primitive(modelBlowupVector))
            modelSX = stackyBlowup(modelSX,cone,modelBlowupVector)
            modelDiv = [0 for i in [1:1:size(getRays(modelX))]]
            newRays = getRays(modelSX.fan)
            blowupVectorIndex = getIndex(newRays, modelBlowupVectorPrim)
            modelDiv[blowupVectorIndex] = 1

            ####################################

            # working on the actual stacky fan
            
            blowupVector = findStackyBarycenter(cone, X)
            blowupVectorPrim = Array{QQFieldElem}(Polymake.common.primitive(blowupVector))
            X = stackyBlowup(X,cone,blowupVector)
            push!(blowupVectors, blowupVectorPrim)
            div[blowupVectorPrim] = 1

            #######################################
        
            modelSH, newD, manual = BerghAmod(modelSX,modelDiv)

            X, div = manualBerghA(X, div,manual)
        end

        for blowupVector in blowupVectors
            X = BerghD(F, div,blowupVectors)
        end
    end
    return X,div
end

X = normal_toric_variety(IncidenceMatrix([[1,2],[4,3]]), [1 0; 1 1; 0 1; -1 0])
F = addStackStructure(X,[1,1,1,1])

# H, divi = BerghC(F,[1,1,1,0]);
# getRays(H.fan)

# divi = Dict(ration([1,2,0])=>0,ration([1,3,0])=> 0, ration([3,0,1])=> 1)

# getRays(BerghE(F,[1,1,0])[1].fan)

# minMaxAggregrate(H,divi)

# simpleste example to test for BerghE is a signle ray, the algorithm will just make it distinguished
# second simplest example is take a cone sigma in Z^k, and then in Z^{k+1}, add a standard basis vector e_{k+1},-e_{k+1}, and the cones are as follows: sigma, sigma with the ray e_{k+1}, sigma with the ray -e_{k+1}
# make e_{k+1},-e_{k+1} distinguished or non distinguished, both cases,

# (a_i b_i) = B (a_i' b_i')

compareDistCones([1,2],[2,3],F,[1,0,1,1])

#look at the picture that dan sent on gmail.
