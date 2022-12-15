//
// Created by Jason Mohoney on 2/28/20.
//

#include "pipeline/trainer.h"

#include "reporting/logger.h"

using std::get;
using std::tie;

PipelineTrainer::PipelineTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config, int logs_per_epoch) {
    dataloader_ = dataloader;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);

    if (model->device_.is_cuda()) {
        pipeline_ = std::make_shared<PipelineGPU>(dataloader, model, true, progress_reporter_, pipeline_config);
    } else {
        pipeline_ = std::make_shared<PipelineCPU>(dataloader, model, true, progress_reporter_, pipeline_config);
    }
}

void PipelineTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }

    dataloader_->initializeBatches(false);

    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        pipeline_->start();
        pipeline_->waitComplete();
        pipeline_->pauseAndFlush();
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);

        if (pipeline_->model_->device_models_.size() > 1) {
            pipeline_->model_->all_reduce();
        }

        dataloader_->nextEpoch();
        progress_reporter_->clear();
        timer.stop();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
}

SynchronousTrainer::SynchronousTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }

    dataloader_->initializeBatches(false);

    Timer timer = Timer(false);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        while (dataloader_->hasNextBatch()) {
            // gets data and parameters for the next batch
            shared_ptr<Batch> batch = dataloader_->getBatch();
            /**
            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                // transfers batch to the GPU
                batch->to(model_->device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }
            */

            /**
            if (batch->node_embeddings_.defined()) {
                batch->node_embeddings_.requires_grad_();
            }
            */

            batch->dense_graph_.performMap();   

            // Directly start updates.
            torch::Tensor src; 
            torch::Tensor dst;
            //if (batch->node_embeddings_.defined()) 
            //{
                SPDLOG_INFO("============Hello!=========");
                src = batch->edges_.select(1, 0);
                dst = batch->edges_.select(1, -1);
            
                // Need to do the right updates on this batch.
                // We directly initialize the gradients here. 
                int sizeOfBatch = src.size(0);
                // Create the gradient matrix, needed??
                // batch->node_gradients_ = torch::zeros(node_embeddings_.sizes());
                // Use efficient accessor:
                auto embeddingAccess = batch->node_embeddings_.accessor<float,2>();
                //auto gradAccess = batch->node_gradients_.accessor<float,2>();
                auto srcAccess = src.accessor<int, 1>();
                auto dstAccess = dst.accessor<int, 1>();
                // compute the updates for pagerank
                for (int i = 0; i < sizeOfBatch; i++) 
                {
                    embeddingAccess[dstAccess[i]][1] += 1e-3;
                    //SPDLOG_INFO("New dst Embedding: {} ", embeddingAccess[dstAccess[i]][1]);
                    // SPDLOG_INFO("Pre src Embedding: {} ", embeddingAccess[srcAccess[i]][0]);
                }
            //} 
            // modify to be pr
            // model_->train_batch(batch);
            // model_->train_pr(batch);

            // transfer gradients and update parameters
            /** Do nothing now
            if (batch->node_embeddings_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                    batch->embeddingsToHost();
                } else {
                    dataloader_->updateEmbeddings(batch, true);
                }

                dataloader_->updateEmbeddings(batch, false);
            }
            */

            batch->clear();

            // notify that the batch has been completed
            dataloader_->finishedBatch();

            // log progress
            progress_reporter_->addResult(batch->batch_size_);
        }
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);

        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();
        timer.stop();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
}
