include("toric_helpers.jl")

function BerghD(F::StackyFan,distDictF::Dict{Vector{QQFieldElem}, Int64}, Dlist::Array{Int64,1})
    X=deepcopy(F)
    distDict = deepcopy(distDictF)
    
    rays=getRays(X.fan)


    #find all rays that we wish to make divisorial indicies along them zero
    divisors = Vector{Vector{QQFieldElem}}([])
    for i in [1:1:size(Dlist,1);]
        if Dlist[i] == 1
            push!(divisors,rays[i])
        end
    end

    
    # for single divisor, excute the algorithm that makes the divisorial index along that divisor 0
    
    for divisor in divisors
        while(true)
            rays=getRays(X.fan)
            # Find the cones with maximal divisorial index in X
            subdivTargetCones=minMaxIndexAlongDiv(X,distDict,divisor)
            # If there arde no such cones, the algorithm terminates
            if subdivTargetCones==nothing
                break
            end
            
            blowupList=Array{Array{QQFieldElem,1},1}[]
            # Iterate through cones with maximal divisorial index
            for cone in subdivTargetCones
                # Add each cone's ray representation to blowupList
                push!(blowupList,newRowMinors(rays, cone))
            end
            for coneDefinedWithRays in blowupList
                coneDefinedWithIndices =Int64[]
                rays=getRays(X.fan)
                for ray in coneDefinedWithRays
                    push!(coneDefinedWithIndices,findall(x->x==ray,rays)[1])
                end
                # Sort indices in ascending order
                coneDefinedWithIndices=sort(coneDefinedWithIndices)
                if size(coneDefinedWithIndices,1)==1
                    distDict[rays[coneDefinedWithIndices[1]]]=1
                else
                    exceptional=makePrimitive(findStackyBarycenter(coneDefinedWithIndices,X))
                    # perform the blowup
                    X=stackyBlowup(X,coneDefinedWithIndices, exceptional)
                    # convert exceptional ray to its primitive form
                    distDict[Vector{QQFieldElem}(exceptional)]=1
                end
            end
        end
    end
    return X, distDict
end









# function BerghD(F::StackyFan,divlist::Array{Int64,1}, Dlist::Array{Int64,1})
#     X=deepcopy(F)
#     slicedRayMatrix=getRays(X.fan)
#     div=Dict{Vector{QQFieldElem}, Int64}()
#     # Populate dictionary of divisorial rays (div) from divlist
#     for i in 1:size(slicedRayMatrix,1)
#         div[slicedRayMatrix[i]]=divlist[i]
#     end
#     divisors = Vector{Vector{QQFieldElem}}([])
#     for i in [1:1:size(Dlist,1);]
#         if Dlist[i] == 1
#             push!(divisors,slicedRayMatrix[i])
#         end
#     end
#     for divisor in divisors
#         while(true)
#             div= deepcopy(div)
#             slicedRayMatrix=getRays(X.fan)
#             # Find the cones with maximal divisorial index in X
#             subdivTargetCones=minMaxIndexAlongDiv(X,div,divisor)
#             # If there arde no such cones, the algorithm terminates
#             if subdivTargetCones==nothing
#                 break
#             end
#             blowupList=Array{Array{QQFieldElem,1},1}[]
#             # Iterate through cones with maximal divisorial index
#             for cone in subdivTargetCones
#                 # Add each cone's ray representation to blowupList
#                 push!(blowupList,newRowMinors(slicedRayMatrix, cone))
#             end
#             for raycone in blowupList
#                 indices=Int64[]
#                 slicedRayMatrix=getRays(X.fan)
#                 for ray in raycone
#                     push!(indices,findall(x->x==ray,slicedRayMatrix)[1])
#                 end
#                 # Sort indices in ascending order
#                 cone=sort(indices)
#                 if size(cone,1)==1
#                     div[slicedRayMatrix[cone[1]]]=1
#                 else
#                     exceptional=findStackyBarycenter(cone,X)
#                     # perform the blowup
#                     X=stackyBlowup(X,cone, exceptional)
#                     # convert exceptional ray to its primitive form
#                     div[Vector{QQFieldElem}(normalize(exceptional))]=1
#                 end
#             end
#         end
#     end
#     return X, div
# end
