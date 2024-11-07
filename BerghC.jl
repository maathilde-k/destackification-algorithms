include("toric_helpers.jl")

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
